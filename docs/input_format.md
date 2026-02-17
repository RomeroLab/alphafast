# Input JSON Format

AlphaFast uses the standard AlphaFold 3 JSON input format. Place one or more `.json` files in your `--input_dir`.

## Field Reference

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Job name. Used for output directory naming. |
| `sequences` | array | Yes | Array of chain definitions (protein, ligand, etc.) |
| `modelSeeds` | array of int | Yes | Random seeds for inference. At least one seed required. |
| `dialect` | string | Yes | Must be `"alphafold3"` |
| `version` | int | Yes | Supported versions: `1`, `2`, `3`, or `4` |

### Optional Fields

| Field | Type | Description |
|-------|------|-------------|
| `bondedAtomPairs` | array | Custom bonded atom pair constraints |
| `userCCD` | string | Custom Chemical Component Dictionary (CCD) entries |

## Protein Chain

A single protein chain with chain ID "A":

```json
{
  "name": "my_protein",
  "sequences": [
    {
      "protein": {
        "id": ["A"],
        "sequence": "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH"
      }
    }
  ],
  "modelSeeds": [42],
  "dialect": "alphafold3",
  "version": 1
}
```

## Homomeric Complex

A homodimer where both chains A and B share the same sequence. List multiple chain IDs in the `id` array:

```json
{
  "name": "my_homodimer",
  "sequences": [
    {
      "protein": {
        "id": ["A", "B"],
        "sequence": "GMRESYANENQFGFKTINSDIHKIVIVGGYGKLGGLFARYLRASGY"
      }
    }
  ],
  "modelSeeds": [1],
  "dialect": "alphafold3",
  "version": 1
}
```

## Heteromeric Complex

Two different protein chains:

```json
{
  "name": "my_heterodimer",
  "sequences": [
    {
      "protein": {
        "id": ["A"],
        "sequence": "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH"
      }
    },
    {
      "protein": {
        "id": ["B"],
        "sequence": "MHSSIVLATVLFVAIASASKTRELCMKSLEHAKVGTSKEAKQDGIDLYKHMFE"
      }
    }
  ],
  "modelSeeds": [1],
  "dialect": "alphafold3",
  "version": 1
}
```

## Protein with Ligand

A protein chain with an ATP ligand, specified using its CCD code:

```json
{
  "name": "protein_atp",
  "sequences": [
    {
      "protein": {
        "id": ["A"],
        "sequence": "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH"
      }
    },
    {
      "ligand": {
        "id": ["B"],
        "ccdCodes": ["ATP"]
      }
    }
  ],
  "modelSeeds": [1],
  "dialect": "alphafold3",
  "version": 1
}
```

## Multiple Seeds

Provide multiple seeds to generate independent structure samples:

```json
{
  "name": "multi_seed",
  "sequences": [
    {
      "protein": {
        "id": ["A"],
        "sequence": "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH"
      }
    }
  ],
  "modelSeeds": [1, 2, 3, 4, 5],
  "dialect": "alphafold3",
  "version": 1
}
```

Alternatively, use the `--num_seeds` flag in `run_alphafold.py` to auto-generate N seeds from a single seed.

## DNA/RNA Chains

DNA and RNA chain types are part of the AlphaFold 3 JSON schema but are not currently supported by AlphaFast. The MSA search for nucleotide chains requires HMMER (Jackhmmer/nhmmer), which is not included in the AlphaFast container. Nucleotide support is planned for a future release.

## Batch Processing

Place multiple JSON files in a single directory and pass it as `--input_dir`:

```
inputs/
  protein_1.json
  protein_2.json
  protein_3.json
```

Each file is processed independently. When `--batch_size` is set, protein sequences from multiple files are grouped into efficient batched MMseqs2-GPU searches.
