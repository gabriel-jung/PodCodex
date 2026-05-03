"""Override the pyinstaller-hooks-contrib hook for PyAV.

The contrib hook calls ``collect_submodules("av")`` and (on Windows)
copies ``av.libs/*.dll`` — both pull in PyAV's GPL FFmpeg. We ship a
stub at ``packaging/pyi_av_stub/av`` instead. See LICENSE_AUDIT.md.
"""

hiddenimports: list[str] = []
binaries: list[tuple[str, str]] = []
datas: list[tuple[str, str]] = []
