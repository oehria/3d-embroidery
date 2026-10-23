# Preview tool

This C++ tool previews the deformed embroidery mesh using ARAP (as-rigid-as-possible) deformation.

## 1. Prepare the results

Run the preparation script from the `preview_tool` directory:

```sh
sh prepare.sh <results_folder> <output_folder>
```

Use the inverse design tool's results folder (`inverse_design/res`) as `<results_folder>`. It should contain `pattern.svg` and `shrinky_out.obj`. For example, starting from the repository root:

```sh
cd preview_tool
sh prepare.sh ../inverse_design/res ./prepared
```

The script creates the preview inputs in `<output_folder>`, including `shell.obj` (the shell mesh) and `lifted.obj` (the stitches lifted onto the target mesh). The script invokes `python`, so make sure it points to a Python environment with the preparation scripts' dependencies installed.

To generate a denser stitch pattern or mesh, edit these parameters in `prepare.sh` before running it:

- `--interval` (currently `0.5`): decrease this value to sample the stitches more densely.
- `--max-area` (currently `0.002`): decrease this value to generate a finer mesh with smaller triangles.

## 2. Build the C++ preview tool

From the `preview_tool` directory, run the usual CMake build pipeline:

```sh
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j
```

## 3. Run the preview

From the `build` directory, run:

```sh
./preview_tool <output_folder>/shell.obj <output_folder>/lifted.obj
```

Paths must be relative to the current working directory or absolute. For the `./prepared` example above, use:

```sh
./preview_tool ../prepared/shell.obj ../prepared/lifted.obj
```

In the preview window:

- Click **One click preview** to run the preview pipeline. Wait for the completion message before inspecting or exporting the result.
- Click **Report shrink error** after running ARAP to report how much the stitches shrank. The terminal shows the average, maximum, and minimum ratios of deformed stitch length to original stitch length; for example, `0.8` means a 20% reduction in length.
- Click **Export obj** to export the deformed mesh as `output.obj` in the current working directory (the `build` directory if you followed the commands above).

## Alternative: standard ARAP in Python

You can also generate the result using standard ARAP with the Python script, without the additional stitching energy provided by the C++ tool. After preparing the results, run this command from the `preview_tool` directory:

```sh
python scripts/arap_deform.py <output_folder>/shell.obj <output_folder>/lifted.obj
```

For the `./prepared` example above:

```sh
python scripts/arap_deform.py ./prepared/shell.obj ./prepared/lifted.obj
```
