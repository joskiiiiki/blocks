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
        python = pkgs.python313;
        pythonPackages = python.pkgs;
        
        # Override pygame to disable failing tests and add numpy
        pygame-ce = pythonPackages.pygame.overridePythonAttrs (old: {
          disabledTests = (old.disabledTests or []) ++ [
            # Window tests fail in headless environment
            "test_mouse_rect"
            "window_test"
          ];
          # Add numpy as a runtime dependency for sndarray
          propagatedBuildInputs = (old.propagatedBuildInputs or []) ++ [
            pythonPackages.numpy
          ];
        });
        
        pygame-gui = pythonPackages.pygame-gui.override {
          inherit pygame-ce; 
        };
      in
      {
        devShells.default = pkgs.mkShell {
          packages = [
            python
            pygame-ce
            pygame-gui
            pythonPackages.numpy
            pythonPackages.pip
            pythonPackages.setuptools
            pythonPackages.platformdirs
            pythonPackages.matplotlib
            pkgs.ty
          ];
          shellHook = ''
            echo "Pygame development environment loaded"
            echo "Python version: $(python --version)"
            echo "Pygame version: $(python -c 'import pygame; print(pygame.version.ver)')"
            echo ""
            echo "Available packages:"
            echo "  - pygame"
            echo "  - pygame_gui"
            echo "  - numpy"
            echo "  - matplotlib"
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
