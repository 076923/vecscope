# Security policy

`vecscope` runs locally, reads the files you point it at, and makes no network connections.
The attack surface is file parsing (`.npy`, `.npz`, `.parquet`, `.jsonl`, `.json`, `.csv`)
and the MCP server, which exposes `audit_embeddings` to any client that can reach it.

## Reporting

Please report vulnerabilities privately through
[GitHub security advisories](https://github.com/076923/vecscope/security/advisories/new)
rather than public issues. Include the input that triggers the problem when you can.

## Notes for operators

- `np.load` is called without `allow_pickle`, so `.npy` files cannot execute code.
- The MCP server reads any path the client passes. Run it with the same trust you would give
  the agent itself.
