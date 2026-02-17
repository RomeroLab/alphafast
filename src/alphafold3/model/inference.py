# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md
# Modifications Copyright 2026 Romero Lab, Duke University

"""Reusable inference core for AlphaFold 3.

Provides ModelRunner, predict_structure, and output writing functions
that can be imported without triggering absl.flags registration.
Used by both run_alphafold.py and queue_worker.py.
"""

from collections.abc import Callable, Sequence
import csv
import dataclasses
import datetime
import functools
import os
import pathlib
import time
from typing import overload

from alphafold3.common import folding_input
from alphafold3.constants import chemical_components
import alphafold3.cpp
from alphafold3.data import featurisation
from alphafold3.data import pipeline
from alphafold3.model import features
from alphafold3.model import model
from alphafold3.model import params
from alphafold3.model import post_processing
from alphafold3.model.components import utils
import haiku as hk
import jax
from jax import numpy as jnp
import numpy as np
import tokamax


def make_model_config(
    *,
    flash_attention_implementation: tokamax.DotProductAttentionImplementation = "triton",
    num_diffusion_samples: int = 5,
    num_recycles: int = 10,
    return_embeddings: bool = False,
    return_distogram: bool = False,
) -> model.Model.Config:
    """Returns a model config with some defaults overridden."""
    config = model.Model.Config()
    config.global_config.flash_attention_implementation = flash_attention_implementation
    config.heads.diffusion.eval.num_samples = num_diffusion_samples
    config.num_recycles = num_recycles
    config.return_embeddings = return_embeddings
    config.return_distogram = return_distogram
    return config


class ModelRunner:
    """Helper class to run structure prediction stages."""

    def __init__(
        self,
        config: model.Model.Config,
        device: jax.Device,
        model_dir: pathlib.Path,
    ):
        self._model_config = config
        self._device = device
        self._model_dir = model_dir

    @functools.cached_property
    def model_params(self) -> hk.Params:
        """Loads model parameters from the model directory."""
        return params.get_model_haiku_params(model_dir=self._model_dir)

    @functools.cached_property
    def _model(
        self,
    ) -> Callable[[jnp.ndarray, features.BatchDict], model.ModelResult]:
        """Loads model parameters and returns a jitted model forward pass."""

        @hk.transform
        def forward_fn(batch):
            return model.Model(self._model_config)(batch)

        return functools.partial(
            jax.jit(forward_fn.apply, device=self._device), self.model_params
        )

    def run_inference(
        self, featurised_example: features.BatchDict, rng_key: jnp.ndarray
    ) -> model.ModelResult:
        """Computes a forward pass of the model on a featurised example."""
        featurised_example = jax.device_put(
            jax.tree_util.tree_map(
                jnp.asarray, utils.remove_invalidly_typed_feats(featurised_example)
            ),
            self._device,
        )

        result = self._model(rng_key, featurised_example)
        result = jax.tree.map(np.asarray, result)
        result = jax.tree.map(
            lambda x: x.astype(jnp.float32) if x.dtype == jnp.bfloat16 else x,
            result,
        )
        result = dict(result)
        identifier = self.model_params["__meta__"]["__identifier__"].tobytes()
        result["__identifier__"] = identifier
        return result

    def extract_inference_results(
        self,
        batch: features.BatchDict,
        result: model.ModelResult,
        target_name: str,
    ) -> list[model.InferenceResult]:
        """Extracts inference results from model outputs."""
        return list(
            model.Model.get_inference_result(
                batch=batch, result=result, target_name=target_name
            )
        )

    def extract_embeddings(
        self, result: model.ModelResult, num_tokens: int
    ) -> dict[str, np.ndarray] | None:
        """Extracts embeddings from model outputs."""
        embeddings = {}
        if "single_embeddings" in result:
            embeddings["single_embeddings"] = result["single_embeddings"][
                :num_tokens
            ].astype(np.float16)
        if "pair_embeddings" in result:
            embeddings["pair_embeddings"] = result["pair_embeddings"][
                :num_tokens, :num_tokens
            ].astype(np.float16)
        return embeddings or None

    def extract_distogram(
        self, result: model.ModelResult, num_tokens: int
    ) -> np.ndarray | None:
        """Extracts distogram from model outputs."""
        if "distogram" not in result["distogram"]:
            return None
        distogram = result["distogram"]["distogram"][:num_tokens, :num_tokens, :]
        return distogram


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ResultsForSeed:
    """Stores the inference results (diffusion samples) for a single seed.

    Attributes:
      seed: The seed used to generate the samples.
      inference_results: The inference results, one per sample.
      full_fold_input: The fold input that must also include the results of
        running the data pipeline - MSA and templates.
      embeddings: The final trunk single and pair embeddings, if requested.
      distogram: The token distance histogram, if requested.
    """

    seed: int
    inference_results: Sequence[model.InferenceResult]
    full_fold_input: folding_input.Input
    embeddings: dict[str, np.ndarray] | None = None
    distogram: np.ndarray | None = None


def predict_structure(
    fold_input: folding_input.Input,
    model_runner: ModelRunner,
    buckets: Sequence[int] | None = None,
    ref_max_modified_date: datetime.date | None = None,
    conformer_max_iterations: int | None = None,
    resolve_msa_overlaps: bool = True,
) -> Sequence[ResultsForSeed]:
    """Runs the full inference pipeline to predict structures for each seed."""

    print(f"Featurising data with {len(fold_input.rng_seeds)} seed(s)...")
    featurisation_start_time = time.time()
    ccd = chemical_components.Ccd(user_ccd=fold_input.user_ccd)
    featurised_examples = featurisation.featurise_input(
        fold_input=fold_input,
        buckets=buckets,
        ccd=ccd,
        verbose=True,
        ref_max_modified_date=ref_max_modified_date,
        conformer_max_iterations=conformer_max_iterations,
        resolve_msa_overlaps=resolve_msa_overlaps,
    )
    print(
        f"Featurising data with {len(fold_input.rng_seeds)} seed(s) took"
        f" {time.time() - featurisation_start_time:.2f} seconds."
    )
    print(
        "Running model inference and extracting output structure samples with"
        f" {len(fold_input.rng_seeds)} seed(s)..."
    )
    all_inference_start_time = time.time()
    all_inference_results = []
    for seed, example in zip(fold_input.rng_seeds, featurised_examples):
        print(f"Running model inference with seed {seed}...")
        inference_start_time = time.time()
        rng_key = jax.random.PRNGKey(seed)
        result = model_runner.run_inference(example, rng_key)
        print(
            f"Running model inference with seed {seed} took"
            f" {time.time() - inference_start_time:.2f} seconds."
        )
        print(f"Extracting inference results with seed {seed}...")
        extract_structures = time.time()
        inference_results = model_runner.extract_inference_results(
            batch=example, result=result, target_name=fold_input.name
        )
        num_tokens = len(inference_results[0].metadata["token_chain_ids"])
        embeddings = model_runner.extract_embeddings(
            result=result, num_tokens=num_tokens
        )
        distogram = model_runner.extract_distogram(result=result, num_tokens=num_tokens)
        print(
            f"Extracting {len(inference_results)} inference samples with"
            f" seed {seed} took {time.time() - extract_structures:.2f} seconds."
        )

        all_inference_results.append(
            ResultsForSeed(
                seed=seed,
                inference_results=inference_results,
                full_fold_input=fold_input,
                embeddings=embeddings,
                distogram=distogram,
            )
        )
    print(
        "Running model inference and extracting output structures with"
        f" {len(fold_input.rng_seeds)} seed(s) took"
        f" {time.time() - all_inference_start_time:.2f} seconds."
    )
    return all_inference_results


def write_fold_input_json(
    fold_input: folding_input.Input,
    output_dir: os.PathLike[str] | str,
) -> None:
    """Writes the input JSON to the output directory."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{fold_input.sanitised_name()}_data.json")
    print(f"Writing model input JSON to {path}")
    with open(path, "wt") as f:
        f.write(fold_input.to_json())


def write_outputs(
    all_inference_results: Sequence[ResultsForSeed],
    output_dir: os.PathLike[str] | str,
    job_name: str,
) -> None:
    """Writes outputs to the specified output directory."""
    ranking_scores = []
    max_ranking_score = None
    max_ranking_result = None

    output_terms = (
        pathlib.Path(alphafold3.cpp.__file__).parent / "OUTPUT_TERMS_OF_USE.md"
    ).read_text()

    os.makedirs(output_dir, exist_ok=True)
    for results_for_seed in all_inference_results:
        seed = results_for_seed.seed
        for sample_idx, result in enumerate(results_for_seed.inference_results):
            sample_dir = os.path.join(output_dir, f"seed-{seed}_sample-{sample_idx}")
            os.makedirs(sample_dir, exist_ok=True)
            post_processing.write_output(
                inference_result=result,
                output_dir=sample_dir,
                name=f"{job_name}_seed-{seed}_sample-{sample_idx}",
            )
            ranking_score = float(result.metadata["ranking_score"])
            ranking_scores.append((seed, sample_idx, ranking_score))
            if max_ranking_score is None or ranking_score > max_ranking_score:
                max_ranking_score = ranking_score
                max_ranking_result = result

        if embeddings := results_for_seed.embeddings:
            embeddings_dir = os.path.join(output_dir, f"seed-{seed}_embeddings")
            os.makedirs(embeddings_dir, exist_ok=True)
            post_processing.write_embeddings(
                embeddings=embeddings,
                output_dir=embeddings_dir,
                name=f"{job_name}_seed-{seed}",
            )

        if (distogram := results_for_seed.distogram) is not None:
            distogram_dir = os.path.join(output_dir, f"seed-{seed}_distogram")
            os.makedirs(distogram_dir, exist_ok=True)
            distogram_path = os.path.join(
                distogram_dir, f"{job_name}_seed-{seed}_distogram.npz"
            )
            with open(distogram_path, "wb") as f:
                np.savez_compressed(f, distogram=distogram.astype(np.float16))

    if max_ranking_result is not None:  # True iff ranking_scores non-empty.
        post_processing.write_output(
            inference_result=max_ranking_result,
            output_dir=output_dir,
            # The output terms of use are the same for all seeds/samples.
            terms_of_use=output_terms,
            name=job_name,
        )
        # Save csv of ranking scores with seeds and sample indices, to allow easier
        # comparison of ranking scores across different runs.
        with open(
            os.path.join(output_dir, f"{job_name}_ranking_scores.csv"), "wt"
        ) as f:
            writer = csv.writer(f)
            writer.writerow(["seed", "sample", "ranking_score"])
            writer.writerows(ranking_scores)


@overload
def process_fold_input(
    fold_input: folding_input.Input,
    data_pipeline_config: pipeline.DataPipelineConfig | None,
    model_runner: None,
    output_dir: os.PathLike[str] | str,
    buckets: Sequence[int] | None = None,
    ref_max_modified_date: datetime.date | None = None,
    conformer_max_iterations: int | None = None,
    resolve_msa_overlaps: bool = True,
    force_output_dir: bool = False,
) -> folding_input.Input: ...


@overload
def process_fold_input(
    fold_input: folding_input.Input,
    data_pipeline_config: pipeline.DataPipelineConfig | None,
    model_runner: ModelRunner,
    output_dir: os.PathLike[str] | str,
    buckets: Sequence[int] | None = None,
    ref_max_modified_date: datetime.date | None = None,
    conformer_max_iterations: int | None = None,
    resolve_msa_overlaps: bool = True,
    force_output_dir: bool = False,
) -> Sequence[ResultsForSeed]: ...


def process_fold_input(
    fold_input: folding_input.Input,
    data_pipeline_config: pipeline.DataPipelineConfig | None,
    model_runner: ModelRunner | None,
    output_dir: os.PathLike[str] | str,
    buckets: Sequence[int] | None = None,
    ref_max_modified_date: datetime.date | None = None,
    conformer_max_iterations: int | None = None,
    resolve_msa_overlaps: bool = True,
    force_output_dir: bool = False,
) -> folding_input.Input | Sequence[ResultsForSeed]:
    """Runs data pipeline and/or inference on a single fold input.

    Args:
      fold_input: Fold input to process.
      data_pipeline_config: Data pipeline config to use. If None, skip the data
        pipeline.
      model_runner: Model runner to use. If None, skip inference.
      output_dir: Output directory to write to.
      buckets: Bucket sizes to pad the data to, to avoid excessive re-compilation
        of the model. If None, calculate the appropriate bucket size from the
        number of tokens. If not None, must be a sequence of at least one integer,
        in strictly increasing order. Will raise an error if the number of tokens
        is more than the largest bucket size.
      ref_max_modified_date: Optional maximum date that controls whether to allow
        use of model coordinates for a chemical component from the CCD if RDKit
        conformer generation fails and the component does not have ideal
        coordinates set. Only for components that have been released before this
        date the model coordinates can be used as a fallback.
      conformer_max_iterations: Optional override for maximum number of iterations
        to run for RDKit conformer search.
      resolve_msa_overlaps: Whether to deduplicate unpaired MSA against paired
        MSA. The default behaviour matches the method described in the AlphaFold 3
        paper. Set this to false if providing custom paired MSA using the unpaired
        MSA field to keep it exactly as is as deduplication against the paired MSA
        could break the manually crafted pairing between MSA sequences.
      force_output_dir: If True, do not create a new output directory even if the
        existing one is non-empty. Instead use the existing output directory and
        potentially overwrite existing files. If False, create a new timestamped
        output directory instead if the existing one is non-empty.

    Returns:
      The processed fold input, or the inference results for each seed.

    Raises:
      ValueError: If the fold input has no chains.
    """
    print(f"\nRunning fold job {fold_input.name}...")

    if not fold_input.chains:
        raise ValueError("Fold input has no chains.")

    if not force_output_dir and os.path.exists(output_dir) and os.listdir(output_dir):
        new_output_dir = (
            f"{output_dir}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        print(
            f"Output will be written in {new_output_dir} since {output_dir} is"
            " non-empty."
        )
        output_dir = new_output_dir
    else:
        print(f"Output will be written in {output_dir}")

    if data_pipeline_config is None:
        print("Skipping data pipeline...")
    else:
        print("Running data pipeline...")
        fold_input = pipeline.DataPipeline(data_pipeline_config).process(fold_input)

    write_fold_input_json(fold_input, output_dir)
    if model_runner is None:
        print("Skipping model inference...")
        output = fold_input
    else:
        print(
            f"Predicting 3D structure for {fold_input.name} with"
            f" {len(fold_input.rng_seeds)} seed(s)..."
        )
        all_inference_results = predict_structure(
            fold_input=fold_input,
            model_runner=model_runner,
            buckets=buckets,
            ref_max_modified_date=ref_max_modified_date,
            conformer_max_iterations=conformer_max_iterations,
            resolve_msa_overlaps=resolve_msa_overlaps,
        )
        print(f"Writing outputs with {len(fold_input.rng_seeds)} seed(s)...")
        write_outputs(
            all_inference_results=all_inference_results,
            output_dir=output_dir,
            job_name=fold_input.sanitised_name(),
        )
        output = all_inference_results

    print(f"Fold job {fold_input.name} done, output written to {output_dir}\n")
    return output
