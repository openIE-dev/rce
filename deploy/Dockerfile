# The rce toolchain as a container: `rce` on PATH, the working directory
# /srv, and the durable log under /data (a volume). Built from this
# checkout with the toolchain spec/checker/rust-toolchain.toml pins.
#
#   docker build -f deploy/Dockerfile -t rce .
#   (--build-arg REGISTRY=mirror.gcr.io/library pulls the same
#   official base images from a mirror)
#   docker run --rm -p 8080:8080 -v rce-data:/data -v "$PWD/programs/network:/srv:ro" \
#     rce host serve /srv/notes_http.rce --log /data/notes --http 0.0.0.0:8080
ARG REGISTRY=docker.io/library
FROM ${REGISTRY}/rust:1-bookworm AS build
WORKDIR /src
COPY . .
RUN cd spec/checker && rustup show && cargo build --release --locked --bin rce

FROM ${REGISTRY}/debian:bookworm-slim
RUN useradd --system --uid 10001 --home /data rce && mkdir -p /data /srv && chown rce /data
COPY --from=build /src/spec/checker/target/release/rce /usr/local/bin/rce
USER rce
WORKDIR /srv
VOLUME /data
# `rce host serve` takes SIGTERM as its stop input (HC-24), so `docker stop`
# ends the loop cleanly with rce as PID 1. TLS peers are pinned by key, so
# the image needs no CA store.
ENTRYPOINT ["rce"]
CMD ["--version"]
