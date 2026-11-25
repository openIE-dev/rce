# Turnkey deploy of the RCE competence catalog (the live substrate service).
# Build context = the monorepo root (so path deps vc/, spec/checker, compute/ resolve).
FROM rust:1.85 AS build
WORKDIR /src
COPY . .
RUN cargo build --release --manifest-path competence-catalog/Cargo.toml --bin catalog-server

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=build /src/target/release/catalog-server /usr/local/bin/catalog-server
EXPOSE 8765
ENV RCE_CATALOG_PORT=8765
CMD ["catalog-server"]