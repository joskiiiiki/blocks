{
  description = "Python development environment with pygame and pygame-gui";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        python = pkgs.python314;
        pythonPackages = python.pkgs;
        pygame = pythonPackages.pygame.override {
          doCheck = false;
        };
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            python
            pythonPackages.numpy
            pythonPackages.pygame-gui
            pythonPackages.pip
            pythonPackages.setuptools
            pythonPackages.platformdirs
            pythonPackages.matplotlib
            pkgs.ty
            pygame
          ];

          shellHook = ''
            echo "Pygame development environment loaded"
            echo "Python version: $(python --version)"
            echo "Pygame version: $(python -c 'import pygame; print(pygame.version.ver)')"
            echo ""
            echo "Available packages:"
            echo "  - pygame"
            echo "  - pygame_gui"
          '';

          # Required for pygame on some systems
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
            pkgs.libGL
            pkgs.libGLU
            pkgs.xorg.libX11
            pkgs.xorg.libXext
            pkgs.xorg.libXrandr
            pkgs.xorg.libXi
          ];
        };
      }
    );
}
