import gymnasium as gym
import numpy as np


env = gym.make('CliffWalking-v0', render_mode="human")

def softmax_convert_into_pi_from_theta(theta):
    beta = 1.0
    [m, n] = theta.shape
    pi = np.zeros((m, n))

    exp_theta = np.exp(beta * theta)

    for i in range(0, m):
        pi[i] = exp_theta[i] / np.nansum(exp_theta[i])
        # softmax로 계산하는 코드

    pi = np.nan_to_num(pi)

    return pi

def get_next_action(pi, observation):
    direction = ['up', 'right', 'down', 'left']
    next_action = np.random.choice(direction, p=pi[observation])

    if next_action == 'up':
        action = 0
    elif next_action == 'right':
        action = 1
    elif next_action == 'down':
        action = 2
    elif next_action == 'left':
        action = 3

    return action

def goal_cliff_return_s_a_history(pi):
    observation, info = env.reset()
    s_a_history = [[observation, np.nan]]
    next_action = get_next_action(pi, observation)

    while True:
        # observation: 위치
        observation, reward, terminated, truncated, info = env.step(next_action)  # 역학 1단계 실행
        if observation < 37:
            next_action = get_next_action(pi, observation)
        else :
            next_action = 37
        s_a_history[-1][1] = next_action

        s_a_history.append([observation, np.nan])

        if terminated or truncated:
            break

    return s_a_history

def update_theta(theta, pi, s_a_history):
    # Policy Gradient Methods for Reinforcement Learning with Function Approximation 참고한 수식
    # (https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf)

    eta = 0.1 # 학습률
    T = len(s_a_history) - 1 # 목표 지점에 이르기까지 단계 수

    [m, n] = theta.shape # theta의 행렬 크기를 구함
    delta_theta = theta.copy()

    for observation in range(0, m):
        for action in range(0, n):
            if not(np.isnan(theta[observation, action])):
                SA_observation = [SA for SA in s_a_history if SA[0] == observation]
                SA_observation_action = [SA for SA in s_a_history if SA == [observation, action]]

                N_observation = len(SA_observation)
                N_observation_action = len(SA_observation_action)

                delta_theta[observation, action] = (N_observation_action - pi[observation, action] * N_observation) / T

    new_theta = theta + eta * delta_theta

    return new_theta

theta = np.array([
    [np.nan, 1, 1, np.nan], [np.nan, 1, 1, 1], [np.nan, 1, 1, 1], [np.nan, 1, 1, 1], [np.nan, 1, 1, 1], [np.nan, 1, 1, 1], [np.nan, 1, 1, 1], [np.nan, 1, 1, 1], [np.nan, 1, 1, 1], [np.nan, 1, 1, 1], [np.nan, 1, 1, 1], [np.nan, np.nan, 1, 1],
    [1, 1, 1, np.nan], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, np.nan, 1, 1],
    [1, 1, 1, np.nan], [1, 1, np.nan, 1], [1, 1, np.nan, 1], [1, 1, np.nan, 1], [1, 1, np.nan, 1], [1, 1, np.nan, 1], [1, 1, np.nan, 1], [1, 1, np.nan, 1], [1, 1, np.nan, 1], [1, 1, np.nan, 1], [1, 1, np.nan, 1], [1, np.nan, 1, 1],
    [1, np.nan, np.nan, np.nan]
])

pi = softmax_convert_into_pi_from_theta(theta)
stop_epsilon = 10**-4

pi_file_path = "pi.txt"

with open(pi_file_path, 'w') as output_file:
    while True:
        s_a_history = goal_cliff_return_s_a_history(pi)
        new_theta = update_theta(theta, pi, s_a_history)
        new_pi = softmax_convert_into_pi_from_theta(new_theta)

        output_file.write('정책 변화: ', np.sum(np.abs(new_pi - pi)))  # 정책의 변화를 출력
        output_file.write('목표 지점에 이르기까지 걸린 단계 수는 ' + str(len(s_a_history) - 1) + '단계입니다')

        theta = new_theta
        pi = new_pi

        if np.sum(np.abs(new_pi - pi)) < stop_epsilon:
            break

    np.set_printoptions(precision=3, suppress=True)
    output_file.write('최종 정책: ', pi)

env.close()
