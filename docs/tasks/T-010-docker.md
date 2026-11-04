# T-010: Dockerfile + CI build-and-run

- **Status:** Done
- **Phase:** M1
- **Modules:** [deploy](../modules/deploy/README.md)
- **Requirements:** REQ-S4
- **Depends on:** T-009
- **ADRs:** [ADR-009](../decisions/ADR-009-docker-image.md)
- **TCs:** TC-010

## Acceptance criteria
- [x] `docker build` succeeds; build fails if the model can't be loaded
- [x] `docker run -v data:/data -v out:/out` writes both files; works with `--network none`
- [x] CI builds and runs the image on synthetic data

## Notes / log
- 2026-11-04: verified in the session container (daemon started manually; proxy CA injected into a scratch build context only). 96.7 %, 0.43 ms/sample, image 655 MB.

## Explain it (code-defense prep)
- **Why slim + pinned pip?** Small, standard, identical versions to what we tested (pickles are version-sensitive).
- **Why single-threaded BLAS?** One-row computations: threads add start-up latency and timing jitter.
- **Likely questions:** *How do staff run it?* → README "Docker" section. *Where do outputs go?* → `/out`, the container's working directory, mounted from the host.
