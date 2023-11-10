# CV2023 - PA2

### Sanha Hwang (20231126)

- 본 폴더의 구성은 다음과 같습니다.
  - output의 각 폴더는 prior result가 있어서 이전의 결과물들을 아카이빙 했습니다.

```
  PA2
  ├── Data
  │ ├── sfm00~14 : input images
  │ └── intrinsic.txt : initial K
  │
  │── result
  │ ├── final : final 3D point
  │ └── 망함1 : bad result
  │ 
  ├── README.md
  │ 
  ├── function, two_view_recon_info : given function and information
  │  
  ├── cv : virtual env
  ├── README.md
  │ 
  ├── rough.py : test codes with 3, 4 images only
  ├── convert_ply.py : convert 3D points to ply
  ├── main.py : main script for PA2
  ├── method.py : implement extract feature, Ransac, triangulation
  └── utils.py : To imporove visuality, make some function
```

- 따라서 본 폴더로 디렉토리를 변경한 후 (./PA2/) 아래의 코드를 스크립트에서 실행하면 됩니다.

```
python main.py
```
