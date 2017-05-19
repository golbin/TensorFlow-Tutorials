# TensorFlow Tutorials

텐서플로우를 기초부터 응용까지 단계별로 연습할 수 있는 소스 코드를 제공합니다.

텐서플로우 공식 사이트에서 제공하는 안내서의 대부분의 내용을 다루고 있으며,
공식 사이트에서 제공하는 소스 코드보다는 훨씬 간략하게 작성하였으므로 쉽게 개념을 익힐 수 있을 것 입니다.
또한, 모든 주석은 한글로(!) 되어 있습니다.

다만, 이론에 대한 깊은 이해와 정확한 구현보다는,
다양한 기법과 모델에 대한 기초적인 개념과 텐서플로우의 기본적인 사용법 학습에 촛점을 두었으므로,
구현이 미흡하게 되어 있는 부분이 많음을 고려해 주세요.

또한, 아름다운 코드를 만들기 보다는 순차적인 흐름으로 이해할 수 있도록 코드와 주석과 만들었음을 참고 해 주시면 감사하겠습니다.

## 요구사항

- TensorFlow > 1.0
- Python 3.6
    - numpy 1.12
    - matplotlib 2.0

※ 아직 DQN과 ChatBot은 수정/확인 중 입니다.

## 간단한 설명

### [01 - TensorFlow Basic](01%20-%20TensorFlow%20Basic)

- [01 - Basic](01%20-%20TensorFlow%20Basic/01%20-%20Basic.py)
  - 텐서플로우의 연산의 개념과 그래프를 실행하는 방법을 익힙니다.
- [02 - Variable](01%20-%20TensorFlow%20Basic/02%20-%20Variable.py)
  - 텐서플로우의 플레이스홀더와 변수의 개념을 익힙니다.
- [03 - Linear Regression](01%20-%20TensorFlow%20Basic/03%20-%20Linear%20Regression.py)
  - 단순한 선형 회귀 모형을 만들어봅니다.

### [02 - Neural Network Basic](02%20-%20Neural%20Network%20Basic)

- [01 - Classification](02%20-%20Neural%20Network%20Basic/01%20-%20Classification.py)
  - 신경망을 구성하여 간단한 분류 모델을 만들어봅니다.
- [02 - Deep NN](02%20-%20Neural%20Network%20Basic/02%20-%20Deep%20NN.py)
  - 여러개의 신경망을 구성하는 방법을 익혀봅니다.
- [03 - Word2Vec](02%20-%20Neural%20Network%20Basic/03%20-%20Word2Vec.py)
  - 자연어 분석에 매우 중요하게 사용되는 Word2Vec 모델을 간단하게 구현해봅니다.

### [03 - TensorBoard, Saver](03%20-%20TensorBoard,%20Saver)

- [01 - Saver](03%20-%20TensorBoard,%20Saver/01%20-%20Saver.py)
  - 학습시킨 모델을 저장하고 재사용하는 방법을 배워봅니다.
- [02 - TensorBoard](03%20-%20TensorBoard,%20Saver/02%20-%20TensorBoard.py)
  - 텐서보드를 이용해 신경망의 구성과 손실값의 변화를 시각적으로 확인해봅니다.
- [03 - TensorBoard #2](03%20-%20TensorBoard,%20Saver/03%20-%20TensorBoard2.py)
  - 텐서보드에 히스토그램을 추가해봅니다.

### [04 - MNIST](04%20-%20MNIST)

- [01 - MNIST](04%20-%20MNIST/01%20-%20MNIST.py)
  - 머신러닝 학습의 Hello World 와 같은 MNIST(손글씨 숫자 인식) 문제를 신경망으로 풀어봅니다.
- [02 - Dropout](04%20-%20MNIST/02%20-%20Dropout.py)
  - 과적합 방지를 위해 많이 사용되는 Dropout 기법을 사용해봅니다.

### [05 - CNN](05%20-%20CNN)

- [01 - CNN](05%20-%20CNN/01%20-%20CNN.py)
  - 이미지 처리 분야에서 가장 유명한 신경망 모델인 CNN 을 이용하여 더 높은 인식률을 만들어봅니다.
- [02 - tf.layers](05%20-%20CNN/02%20-%20tf.layers.py)
  - 신경망 구성을 손쉽게 해 주는 High level API 인 layers 를 사용해봅니다.

### [06 - Autoencoder, GAN](06%20-%20Autoencoder)

- [01 - Autoencoder](06%20-%20Autoencoder/01%20-%20Autoencoder.py)
  - 대표적인 비감독(Unsupervised) 학습 방법인 Autoencoder 를 사용해봅니다.

### [07 - GAN](07%20-%20GAN)

- [01 - GAN](07%20-%20GAN/01%20-%20GAN.py)
  - 2016년에 가장 관심을 많이 받았던 비감독 학습 방법인 GAN 을 구현해봅니다.
- [02 - GAN #2](07%20-%20GAN/02%20-%20GAN2.py)
  - GAN 을 응용하여 원하는 숫자의 손글씨 이미지를 생성하는 모델을 만들어봅니다. 이런 방식으로 흑백 사진을 컬러로 만든다든가, 또는 선화를 채색한다든가 하는 응용이 가능합니다.

### [08 - RNN](08%20-%20RNN)

- [01 - Counting](08%20-%20RNN/01%20-%20MNIST.py)
  - 자연어 처리나 음성 처리 분야에 많이 사용되는 RNN 의 기본적인 사용법을 익힙니다.
- [02 - Counting](08%20-%20RNN/02%20-%20Counting.py)
  - 순서가 있는 데이터에 강한 RNN 특징을 이용해 숫자를 순서대로 세는 모델을 구현해봅니다.
- [03 - Seq2Seq](08%20-%20RNN/03%20-%20Seq2Seq.py)
  - 챗봇, 번역, 이미지 캡셔닝등에 사용되는 시퀀스 학습/생성 모델인 Seq2Seq 을 구현해봅니다.
- [Chatbot](08%20-%20RNN/ChatBot)
  - Seq2Seq 모델을 이용해 간단한 챗봇을 만들어봅니다.

### [09 - Inception](09%20-%20Inception)

구글에서 개발한 이미지 인식에 매우 뛰어난 신경망 모델인 Inception 을 사용해봅니다.

신경망 모델을 직접 구현할 필요 없이, 간단한 스크립트 작성만으로 자신만의 데이터를 이용해 매우 뛰어난 인식률을 가진 프로그램을 곧바로 실무에 적용할 수 있습니다.

자세한 내용은 [09 - Inception/README.md](09%20-%20Inception/README.md) 문서를 참고 해 주세요.

### [10 - DQN](10%20-%20DQN)

알파고로 유명한 구글의 딥마인드에서 개발한 딥러닝을 이용한 강화학습인 DQN 을 구현해봅니다.

조금 복잡해보이지만, 핵심적인 부분을 최대한 분리해두었으니 충분히 따라가실 수 있을 것 입니다.

자세한 내용은 [10 - DQN/README.md](10%20-%20DQN/README.md) 문서를 참고 해 주세요.

## 참고

조금 더 기초적인 이론에 대한 내용은 다음 강좌와 저장소를 참고하세요.

- [모두를 위한 머신러닝/딥러닝 강의](https://www.youtube.com/watch?v=BS6O0zOGX4E&list=PLlMkM4tgfjnLSOjrEJN31gZATbcj_MpUm) (홍콩 과기대 김성훈 교수님 강좌)
- [강좌 실습 코드](https://github.com/golbin/TensorFlow-ML-Exercises) (내가 만듬)
