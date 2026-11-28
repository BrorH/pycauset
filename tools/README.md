# tools/

`build.ps1`: a thin wrapper around the pip build for Windows (`documentation/internals/Build System.md`).
`release.ps1`: creates and pushes the next version tag (`documentation/project/protocols/Release Process.md`).
`make_gallery.py`: regenerates the plot images the guides show (`documentation/docs/assets/gallery/`; needs plotly and kaleido).
Checks that used to live here are tests now (`tests/engine/test_api_surface.py`).
