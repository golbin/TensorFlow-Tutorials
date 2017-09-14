# Inception 모델 사용해보기

구글에서 제공하는 높은 성능의 이미지 처리 신경망인 Inception 을 사용해봅니다.

제공되는 스크립트와 데이터로 매우 쉽게, 그리고 실무에도 바로 사용해 볼 수 있습니다.

### 학습시켜 볼 샘플 자료 다운로드

```
curl http://download.tensorflow.org/example_images/flower_photos.tgz | tar xz -C ./workspace
```

자신이 가진 다른 이미지를 학습시켜보고 싶다면, 학습시킬 사진을 각각의 레이블 이름으로 폴더를 생성하고, 그 안에 폴더 이름에 맞는 사진을 넣어두면 됩니다.

### 학습 실행

```
# python retrain.py \
    --bottleneck_dir=./workspace/bottlenecks \
    --model_dir=./workspace/inception \
    --output_graph=./workspace/flowers_graph.pb \
    --output_labels=./workspace/flowers_labels.txt \
    --image_dir ./workspace/flower_photos \
    --how_many_training_steps 1000
```

### 추론 테스트

```
# python predict.py ./workspace/flower_photos/roses/20409866779_ac473f55e0_m.jpg
```

### retrain.py 주요 옵션

- --bottleneck_dir : 학습할 사진을 인셉션 용으로 변환해서 저장할 폴더
- --model_dir : inception 모델을 다운로드 할 경로
- --image_dir : 원본 이미지 경로
- --output_graph : 추론에 사용할 학습된 파일(.pb) 경로
- --output_labels : 추론에 사용할 레이블 파일 경로
- --how_many_training_steps : 얼만큼 반복 학습시킬 것인지

### 참고

- [텐서플로 모델 저장소](https://github.com/tensorflow/models)에서 더 많은 모델들을 다운로드 받고 시험해 볼 수 있습니다
- [텐서플로 저장소의 retrain.py 원본 위치](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/image_retraining)
