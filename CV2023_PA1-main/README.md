## PA1 : Multi Image Denoising

### 황산하 (20231126), GIST AI대학원

- 본 폴더의 구성은 다음과 같습니다.
  - output의 각 폴더는 prior result가 있어서 이전의 결과물들을 아카이빙 했습니다. 
```
  CV2023_PA1-MAIN
  ├── input, other_input : input images
  │
  │── output
  │ ├── Cost : MSE & PSNR
  │ ├── Final_Disparity : Final Disparity (array)
  │ ├── Final_warped_image : Final warped image (result)
  │ ├── Warped_images : Warped_images (png)
  │ ├── Intermediate_Disparity : Disparities of Each images (input)  
  │ └── agg_warped_image*{date}.png # final output
  │  
  ├── README.md
  ├── target
  │   └── gt.png # Ground Truth image
  ├── _init_.py
  ├── aggregate_cost_volume : Semi global matching
  ├── Semi_Global_Matching.py : Main code for getting final image
  ├── similarity.py : calculate disparity (SAD, SSD)
  ├── test.py : roughly test some function
  └── warp.py : Image warping code
```
- 따라서 본 폴더로 디렉토리를 변경한 후 (./CV2023_PA1-MAIN/) 아래의 코드를 스크립트에서 실행하면 됩니다.

```
python Semi_Global_Matching.py
```
