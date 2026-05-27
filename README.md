# ted-sfc

Identifying and analyzing traffic events in large-scale, unstructured video data from vehicle-mounted cameras is a significant challenge for enhancing advanced driver assistance systems (ADAS). This thesis presents a conceptual framework that leverages machine learning (ML), optical flow (OF), and object detection for efficient traffic event detection, utilizing space-filling curves (SFCs) to reduce data dimensionality. The project uses the real-world Zenseact Open Dataset (ZOD) and Waymo Open Dataset for dataset experiments. This framework could serve as a foundation for scalable solutions to analyze large volumes of unstructured data in the form of traffic event detection or other contexts.

## Environment

Use the Conda environment described below as the baseline project setup. The project expects Python 3.9, FFmpeg, PyTorch/torchvision, OpenCV, PyAV, NumPy, pandas, PyYAML, and the packages listed in `requirements.txt`.

The object-detection methods use the newer torchvision weights API, including `FasterRCNN_ResNet50_FPN_Weights`, so the PyTorch/torchvision versions need to support that API. The setup used for these experiments was tested with `torch==2.8.0` and `torchvision==0.23.0`.

```bash
conda activate pyTED
python --version
python -m pip check
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import cv2; print(cv2.__version__, cv2.cuda.getCudaEnabledDeviceCount())"
ffmpeg -version
```

Local datasets, generated outputs, and cached model weights should stay outside git.

## Running TED-SFC

### 1. Prerequisites

- [Conda](https://docs.anaconda.com/free/miniconda/index.html)
- [FFmpeg](https://ffmpeg.org/download.html)
- NVIDIA CUDA-enabled GPU (optional)

Note: While the pipeline does support CPU, it has only been tested with NVIDIA CUDA-enabled GPUs.

### 2. Create environment

```bash
conda create -n pyTED python=3.9
conda activate pyTED
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

For CPU-only machines or a different CUDA version, install a matching PyTorch/torchvision build before running `pip install -r requirements.txt`.

The saliency methods expect their pretrained weights under `models/saliency/`:

- `models/saliency/mlnet_25.pth`
- `models/saliency/tasednet_iter_1000.pt`
- `models/saliency/TranSalNet_Res.pth`
- `models/saliency/resnet50-0676ba61.pth`

### 3. Prepare data

Make sure every dataset follows the structure below. The pipeline expects each video folder name to match the `.avi` file inside that folder.

```txt
data/
  [dataset_name]/
    [video_id]/
      [video_id].avi
    ...
```

Scripts for processing datasets into the correct structure are provided in `src/scripts`. The project currently supports the following datasets:

- [ZOD](https://www.zod.zenseact.com)
- [Waymo Open Dataset](https://waymo.com/open/)

ZOD input should contain the original `sequences/` directory. The ZOD processing script converts each selected `camera_front_blur` frame sequence into an AVI file.

```bash
python src/scripts/zod/process.py path/to/original/dataset data/ZOD1 --mode random --nr-videos 10
```

Resulting ZOD structure:

```txt
data/ZOD1/
  000011/
    000011.avi
  000046/
    000046.avi
```

Waymo input should contain `.tfrecord` segment files. The Waymo helper extracts camera JPEG frames from TFRecords without TensorFlow. It requires the Waymo protobuf package under `vendor/waymo/wheel`.

```bash
python src/scripts/waymo/extract_tfrecord_frames_no_tf.py path/to/waymo/tfrecords data/waymo_frames --camera FRONT --max-frames 200
```

After extraction, convert each selected Waymo frame folder to an AVI and place it in the same folder/file layout used by ZOD:

```txt
data/Waymo1/
  000012/
    000012.avi
  000015/
    000015.avi
```

For example:

```bash
mkdir -p data/Waymo1/000012
ffmpeg -framerate 10 -i data/waymo_frames/[segment_id]/%06d.jpg -vf scale=1280:720 -c:v libx264 -crf 23 -preset veryfast data/Waymo1/000012/000012.avi
```

The Waymo subset IDs used for the experiments are listed in the dataset table below.

#### Automatic Pedestrian Sequence Filtering

A YOLO-based filtering script was used to automatically scan ZOD and Waymo dataset sequences and identify frames containing pedestrians. This step helped reduce manual inspection effort by generating a shortlist of candidate sequences likely to contain pedestrian-related events.

The script processes frames from each sequence, performs pedestrian detection using a configurable confidence threshold, and generates CSV reports containing matched sequences and matched frames.

##### Script location (in the repo):

```bash
scripts/find_pedestrian_sequences.py
```

##### To run the script:

```bash
python find_pedestrian_sequences.py \
  --dataset_root /path/to/dataset/sequences \
  --start_sequence 000001 \
  --end_sequence 001472 \
  --person_conf 0.40 \
  --output_prefix zod_pedestrian
```

##### Example Output Files:

```bash
zod_pedestrian_sequences_report.csv
zod_pedestrian_frames_report.csv
waymo_pedestrian_sequences_report.csv
waymo_pedestrian_frames_report.csv
```

The generated reports were later used for manual visual analysis and final positive/negative sequence selection for the thesis experiments.

**Important**: Dataset conversion requires the `ffmpeg` executable. If `ffmpeg -version` does not work inside the Conda environment, run the conversion command from a shell where FFmpeg is available.

### 4. Optional: Enable GPU-accelerated Optical Flow (NVIDIA CUDA-enabled GPUs only)

To enable running the optical flow model on the GPU, compile opencv from source with the cudaoptflow module:

1. Create a new conda environment, `pyTED-cuda-cv`, using the instructions from [Step 2](#2-create-environment)
1. Uninstall the current version of opencv `pip uninstall opencv-python`
1. Follow [this guide](https://danielhavir.com/notes/install-opencv/) by Daniel Havir. Note:
   - Get the latest versions of opencv and opencv_contrib from the official repositories: [opencv](https://github.com/opencv/opencv/releases) and [opencv_contrib](https://github.com/opencv/opencv_contrib/tags)
   - Use the newly created environment `pyTED-cuda-cv` instead of `cv`
   - Replace references to python3.6 with python3.9
   - Ensure all the environment variables are correctly defined before running the cmake command. Example values:
     - `$python_exec: /path/to/miniconda3/envs/pyTED-cuda-cv/bin/python`
     - `$include_dir: /path/to/miniconda3/envs/pyTED-cuda-cv/include/python3.9`
     - `$library: /path/to/miniconda3/envs/pyTED-cuda-cv/lib/libpython3.9.so`
     - `$default_exec: /path/to/miniconda3/envs/pyTED-cuda-cv/bin/python3.9`
1. Test your installation with `python -c "import cv2; print('CUDA is available:', cv2.cuda.getCudaEnabledDeviceCount() > 0)"`
   - If you get an error about GCC version 12.0.0 being required, run `conda install conda-forge::libgcc-ng==12`

Only use the `pyTED-cuda-cv` environment when running the optical flow model.

### 5. Run the processing pipeline

To run the TED-SFC pipeline, run the following command:

```bash
python src/pipeline.py -d path/to/dataset -o path/to/output -c path/to/config.yml -m [mlnet | tasednet | transalnet | optical-flow | faster-rcnn | detr] [--cpu] [--annotations-path=path/to/annotations]
```

The pipeline will extract features from the videos using the selected method, convert cell values to Morton codes, and run the event detection. The results will be placed in the output directory. Evaluation will be run if the annotations path is provided.

The `faster-rcnn` and `detr` methods add object-detection feature extraction before the original Morton-code event detector. They detect COCO `person` objects, map the strongest detection confidence per frame to the existing grid, and then continue through the same SFC pipeline as the original methods. The first run may download pretrained model weights through PyTorch or Torch Hub.

ZOD example:

```bash
python src/pipeline.py -d data/ZOD1 -o outputs/zod-faster-rcnn/ZOD1 -c config/zod/pedestrian_crossing.yml -m faster-rcnn --annotations-path annotations/zod_pedestrian.yml
```

Waymo example:

```bash
python src/pipeline.py -d data/Waymo1 -o outputs/waymo-faster-rcnn/Waymo1 -c config/waymo/pedestrian_crossing.yml -m faster-rcnn --annotations-path annotations/waymo_pedestrian.yml
```

For Waymo MLNet/attention experiments, `config/waymo/pedestrian_crossing_tuned.yml` contains a tuned detector configuration.

### 6. Evaluate

To evaluate the event detection in terms of F1-score, sensitivity, specificity and mean IoU, pass the output directory that contains `event_window.csv`:

`python src/evaluate.py path/to/output path/to/annotations.yml`

## Datasets

### ZOD

| ZOD Positives | ZOD (1) Negatives | ZOD (2) Negatives | ZOD (3) Negatives | ZOD (4) Negatives |
| ------------- | ----------------- | ----------------- | ----------------- | ----------------- |
| 000011        | 000082            | 000007            | 000143            | 000024            |
| 000046        | 000084            | 000019            | 000217            | 000168            |
| 000113        | 000098            | 000236            | 000229            | 000231            |
| 000169        | 000137            | 000238            | 000230            | 000234            |
| 000237        | 000161            | 000390            | 000232            | 000411            |
| 000292        | 000162            | 000414            | 000296            | 000464            |
| 000314        | 000306            | 000530            | 000461            | 000614            |
| 000316        | 000327            | 000583            | 000541            | 000680            |
| 000383        | 000684            | 000869            | 000603            | 000705            |
| 000389        | 000864            | 000905            | 000871            | 000877            |
| 000398        | 000865            | 000935            | 000881            | 000880            |
| 000433        | 000870            | 001012            | 000956            | 001049            |
| 000521        | 000900            | 001091            | 000977            | 001300            |
| 000653        | 000934            | 001199            | 001011            | 001307            |
| 000860        | 001326            | 001273            | 001067            | 001328            |
| 000893        | 001352            | 001294            | 001200            | 001341            |
|               | 001457            | 001245            | 001295            | 001412            |

### Waymo

| Waymo Positives | Waymo (1) Negatives | Waymo (2) Negatives | Waymo (3) Negatives | Waymo (4) Negatives |
| --------------- | ------------------- | ------------------- | ------------------- | ------------------- |
| 000012          | 000001              | 000026              | 000050              | 000090              |
| 000015          | 000004              | 000027              | 000051              | 000093              |
| 000063          | 000005              | 000028              | 000052              | 000097              |
| 000065          | 000006              | 000029              | 000053              | 000098              |
| 000066          | 000007              | 000030              | 000055              | 000107              |
| 000074          | 000008              | 000031              | 000056              | 000109              |
| 000083          | 000009              | 000032              | 000057              | 000118              |
| 000092          | 000013              | 000033              | 000058              | 000153              |
| 000099          | 000014              | 000034              | 000067              | 000163              |
| 000101          | 000016              | 000037              | 000068              | 000166              |
| 000144          | 000019              | 000038              | 000069              | 000167              |
| 000146          | 000020              | 000041              | 000072              | 000169              |
| 000164          | 000021              | 000042              | 000075              | 000170              |
| 000165          | 000022              | 000043              | 000076              | 000175              |
| 000174          | 000023              | 000045              | 000077              | 000182              |
| 000189          | 000024              | 000047              | 000084              | 000201              |
|                 | 000025              | 000048              | 000087              | 000202              |

## Dataset Licensing Information Notice

For ZOD, Zenseact AB has taken all reasonable measures to remove all personally identifiable information, including faces and license plates. To the extent that you like to request removal of specific images from the dataset, please contact privacy@zenseact.com.

For Waymo, use of the Waymo Open Dataset is governed by the [Waymo Open Dataset terms](https://waymo.com/open/terms/). Waymo states that personally identifiable information, including faces and license plates, is removed or hidden with reasonable care.
