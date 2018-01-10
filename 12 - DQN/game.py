# 장애물 회피 게임 즉, 자율주행차:-D 게임을 구현합니다.
import numpy as np
import random

import matplotlib.pyplot as plt
import matplotlib.patches as patches


class Game:
    def __init__(self, screen_width, screen_height, show_game=True):
        self.screen_width = screen_width
        self.screen_height = screen_height
        # 도로의 크기는 스크린의 반으로 정하며, 도로의 좌측 우측의 여백을 계산해둡니다.
        self.road_width = int(screen_width / 2)
        self.road_left = int(self.road_width / 2 + 1)
        self.road_right = int(self.road_left + self.road_width - 1)

        # 자동차와 장애물의 초기 위치와, 장애물 각각의 속도를 정합니다.
        self.car = {"col": 0, "row": 2}
        self.block = [
            {"col": 0, "row": 0, "speed": 1},
            {"col": 0, "row": 0, "speed": 2},
        ]

        self.total_reward = 0.
        self.current_reward = 0.
        self.total_game = 0
        self.show_game = show_game

        if show_game:
            self.fig, self.axis = self._prepare_display()

    def _prepare_display(self):
        """게임을 화면에 보여주기 위해 matplotlib 으로 출력할 화면을 설정합니다."""
        fig, axis = plt.subplots(figsize=(4, 6))
        fig.set_size_inches(4, 6)
        # 화면을 닫으면 프로그램을 종료합니다.
        fig.canvas.mpl_connect('close_event', exit)
        plt.axis((0, self.screen_width, 0, self.screen_height))
        plt.tick_params(top='off', right='off',
                        left='off', labelleft='off',
                        bottom='off', labelbottom='off')

        plt.draw()
        # 게임을 진행하며 화면을 업데이트 할 수 있도록 interactive 모드로 설정합니다.
        plt.ion()
        plt.show()

        return fig, axis

    def _get_state(self):
        """게임의 상태를 가져옵니다.

        게임의 상태는 screen_width x screen_height 크기로 각 위치에 대한 상태값을 가지고 있으며,
        빈 공간인 경우에는 0, 사물이 있는 경우에는 1이 들어있는 1차원 배열입니다.
        계산의 편의성을 위해 2차원 -> 1차원으로 변환하여 사용합니다.
        """
        state = np.zeros((self.screen_width, self.screen_height))

        state[self.car["col"], self.car["row"]] = 1

        if self.block[0]["row"] < self.screen_height:
            state[self.block[0]["col"], self.block[0]["row"]] = 1

        if self.block[1]["row"] < self.screen_height:
            state[self.block[1]["col"], self.block[1]["row"]] = 1

        return state

    def _draw_screen(self):
        title = " Avg. Reward: %d Reward: %d Total Game: %d" % (
                        self.total_reward / self.total_game,
                        self.current_reward,
                        self.total_game)

        # self.axis.clear()
        self.axis.set_title(title, fontsize=12)

        road = patches.Rectangle((self.road_left - 1, 0),
                                 self.road_width + 1, self.screen_height,
                                 linewidth=0, facecolor="#333333")
        # 자동차, 장애물들을 1x1 크기의 정사각형으로 그리도록하며, 좌표를 기준으로 중앙에 위치시킵니다.
        # 자동차의 경우에는 장애물과 충돌시 확인이 가능하도록 0.5만큼 아래쪽으로 이동하여 그립니다.
        car = patches.Rectangle((self.car["col"] - 0.5, self.car["row"] - 0.5),
                                1, 1,
                                linewidth=0, facecolor="#00FF00")
        block1 = patches.Rectangle((self.block[0]["col"] - 0.5, self.block[0]["row"]),
                                   1, 1,
                                   linewidth=0, facecolor="#0000FF")
        block2 = patches.Rectangle((self.block[1]["col"] - 0.5, self.block[1]["row"]),
                                   1, 1,
                                   linewidth=0, facecolor="#FF0000")

        self.axis.add_patch(road)
        self.axis.add_patch(car)
        self.axis.add_patch(block1)
        self.axis.add_patch(block2)

        self.fig.canvas.draw()
        # 게임의 다음 단계 진행을 위해 matplot 의 이벤트 루프를 잠시 멈춥니다.
        plt.pause(0.0001)

    def reset(self):
        """자동차, 장애물의 위치와 보상값들을 초기화합니다."""
        self.current_reward = 0
        self.total_game += 1

        self.car["col"] = int(self.screen_width / 2)

        self.block[0]["col"] = random.randrange(self.road_left, self.road_right + 1)
        self.block[0]["row"] = 0
        self.block[1]["col"] = random.randrange(self.road_left, self.road_right + 1)
        self.block[1]["row"] = 0

        self._update_block()

        return self._get_state()

    def _update_car(self, move):
        """액션에 따라 자동차를 이동시킵니다.

        자동차 위치 제한을 도로가 아니라 화면의 좌우측 끝으로 하고,
        도로를 넘어가면 패널티를 주도록 학습해서 도로를 넘지 않게 만들면 더욱 좋을 것 같습니다.
        """

        # 자동차의 위치가 도로의 좌측을 넘지 않도록 합니다: max(0, move) > 0
        self.car["col"] = max(self.road_left, self.car["col"] + move)
        # 자동차의 위치가 도로의 우측을 넘지 않도록 합니다.: min(max, screen_width) < screen_width
        self.car["col"] = min(self.car["col"], self.road_right)

    def _update_block(self):
        """장애물을 이동시킵니다.

        장애물이 화면 내에 있는 경우는 각각의 속도에 따라 위치 변경을,
        화면을 벗어난 경우에는 다시 방해를 시작하도록 재설정을 합니다.
        """
        reward = 0

        if self.block[0]["row"] > 0:
            self.block[0]["row"] -= self.block[0]["speed"]
        else:
            self.block[0]["col"] = random.randrange(self.road_left, self.road_right + 1)
            self.block[0]["row"] = self.screen_height
            reward += 1

        if self.block[1]["row"] > 0:
            self.block[1]["row"] -= self.block[1]["speed"]
        else:
            self.block[1]["col"] = random.randrange(self.road_left, self.road_right + 1)
            self.block[1]["row"] = self.screen_height
            reward += 1

        return reward

    def _is_gameover(self):
        # 장애물과 자동차가 충돌했는지를 파악합니다.
        # 사각형 박스의 충돌을 체크하는 것이 아니라 좌표를 체크하는 것이어서 화면에는 약간 다르게 보일 수 있습니다.
        if ((self.car["col"] == self.block[0]["col"] and
             self.car["row"] == self.block[0]["row"]) or
            (self.car["col"] == self.block[1]["col"] and
             self.car["row"] == self.block[1]["row"])):

            self.total_reward += self.current_reward

            return True
        else:
            return False

    def step(self, action):
        # action: 0: 좌, 1: 유지, 2: 우
        # action - 1 을 하여, 좌표를 액션이 0 일 경우 -1 만큼, 2 일 경우 1 만큼 옮깁니다.
        self._update_car(action - 1)
        # 장애물을 이동시킵니다. 장애물이 자동차에 충돌하지 않고 화면을 모두 지나가면 보상을 얻습니다.
        escape_reward = self._update_block()
        # 움직임이 적을 경우에도 보상을 줘서 안정적으로 이동하는 것 처럼 보이게 만듭니다.
        stable_reward = 1. / self.screen_height if action == 1 else 0
        # 게임이 종료됐는지를 판단합니다. 자동차와 장애물이 충돌했는지를 파악합니다.
        gameover = self._is_gameover()

        if gameover:
            # 장애물에 충돌한 경우 -2점을 보상으로 줍니다. 장애물이 두 개이기 때문입니다.
            # 장애물을 회피했을 때 보상을 주지 않고, 충돌한 경우에만 -1점을 주어도 됩니다.
            reward = -2
        else:
            reward = escape_reward + stable_reward
            self.current_reward += reward

        if self.show_game:
            self._draw_screen()

        return self._get_state(), reward, gameover
