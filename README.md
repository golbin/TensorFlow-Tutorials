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

- TensorFlow 0.12
- Python 2.7
    - numpy 1.11
    - mathplot 1.5

TensorFlow 1.0 이 출시되면 Python 3.x 와 TensorFlow 1.0 으로 업데이트 할 예정입니다.

## 간단한 설명

### [01 - TensorFlow Basic](./01 - TensorFlow Basic)

- [01 - Basic](./01 - TensorFlow Basic/01 - Basic.py)
  - 텐서플로우의 플레이스홀더와 변수, 연산의 개념과 그래프를 실행하는 방법을 익힙니다.
- [02 - Linear Regression](./01 - TensorFlow Basic/02 - Linear Regression.py)
  - 단순한 선형 회귀 모형을 만들어봅니다.

### [02 - Neural Network Basic](./02 - Neural Network Basic)

- [01 - Classification](./02 - Neural Network Basic/01 - Classification.py)
  - 신경망을 구성하여 간단한 분류 모델을 만들어봅니다.
- [02 - Deep NN](./02 - Neural Network Basic/02 - Deep NN.py)
  - 여러개의 신경망을 구성하는 방법을 익혀봅니다.
- [03 - TensorBoard](./02 - Neural Network Basic/03 - TensorBoard.py)
  - 텐서보드를 이용해 신경망의 구성과 손실값의 변화를 시각적으로 확인해봅니다.
- [04 - Word2Vec](./02 - Neural Network Basic/04 - Word2Vec.py)
  - 자연어 분석에 매주 중요하게 사용되는 Word2Vec 모델을 간단하게 구현해봅니다.

### [03 - MNIST (CNN, Autoencoder)](./03 - MNIST (CNN, Autoencoder))

- [01 - MNIST](./03 - MNIST (CNN, Autoencoder)/01 - MNIST.py)
  - 머신러닝 학습의 Hello World 와 같은 MNIST(손글씨 숫자 인식) 문제를 신경망으로 풀어봅니다.
- [02 - Dropout](./03 - MNIST (CNN, Autoencoder)/02 - Dropout.py)
  - 과적합 방지를 위해 많이 사용되는 Dropout 기법을 사용해봅니다.
- [03 - CNN](./03 - MNIST (CNN, Autoencoder)/03 - CNN.py)
  - 이미지 처리 분야에서 가장 유명한 신경망 모델인 CNN 을 이용하여 더 높은 인식률을 만들어봅니다.
- [04 - Autoencoder](./03 - MNIST (CNN, Autoencoder)/04 - Autoencoder.py)
  - 대표적인 비감독(Unsupervised) 학습 방법인 Autoencoder 를 사용해봅니다.

### [04 - ChatBot (RNN)](./04 - ChatBot (RNN))

- [01 - Counting](./04 - ChatBot (RNN)/01 - Counting.py)
  - 자연어 처리나 음성 처리 분야에 많이 사용되는 RNN 의 기본적인 사용법을 익힙니다.
- [02 - Dynamic RNN](./04 - ChatBot (RNN)/02 - Dynamic RNN.py)
  - 다중 레이어의 RNN 과 더 효율적인 RNN 학습을 위해 텐서플로우에서 제공하는 Dynamic RNN 을 사용해봅니다.
- [03 - Seq2Seq](./04 - ChatBot (RNN)/03 - Seq2Seq.py)
  - 챗봇, 번역, 이미지 캡셔닝등에 사용되는 시퀀스 학습/생성 모델인 Seq2Seq 을 구현해봅니다.
- [04 - Chatbot](./04 - ChatBot (RNN)/04 - ChatBot)
  - Seq2Seq 모델을 이용해 간단한 챗봇을 만들어봅니다.

### [05 - Inception](./05 - Inception)

구글에서 개발한 이미지 인식에 매우 뛰어난 신경망 모델인 Inception 을 사용해봅니다.

신경망 모델을 직접 구현할 필요 없이, 간단한 스크립트 작성만으로 자신만의 데이터를 이용해 매우 뛰어난 인식률을 가진 프로그램을 곧바로 실무에 적용할 수 있습니다.

자세한 내용은 [05 - Inception/README.md](./05 - Inception/README.md) 설명을 참고 해 주세요.

### TODO

- [ ] 강화학습, DQN
- [ ] GAN

## 참고

조금 더 기초적인 이론에 대한 내용은 다음 강좌와 저장소를 참고하세요.

- [모두를 위한 머신러닝/딥러닝 강의](https://www.youtube.com/watch?v=BS6O0zOGX4E&list=PLlMkM4tgfjnLSOjrEJN31gZATbcj_MpUm) (홍콩 과기대 김성훈 교수님 강좌)
- [강좌 실습 코드](https://github.com/golbin/TensorFlow-ML-Exercises) (내가 만듬)
