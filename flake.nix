{
  description = "";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    tinygrad.url = "github:wozeparrot/tinygrad-nix";
    nixpkgs-esp-dev.url = "github:mirrexagon/nixpkgs-esp-dev";
  };

  outputs = inputs @ {
    nixpkgs,
    flake-utils,
    ...
  }:
    flake-utils.lib.eachDefaultSystem
    (
      system: let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [
            inputs.tinygrad.overlays.default
          ];
        };
        nixpkgs-esp-dev = inputs.nixpkgs-esp-dev.packages.${system};
      in {
        devShell = pkgs.mkShell {
          packages = let
            python-packages = p:
              with p; [
                tinygrad
              ];
            python = pkgs.python311;
          in
            with pkgs; [
              llvmPackages_latest.clang
              llvmPackages_latest.openmp

              (python.withPackages python-packages)
              nixpkgs-esp-dev.esp-idf-full
            ];
        };
      }
    );
}
