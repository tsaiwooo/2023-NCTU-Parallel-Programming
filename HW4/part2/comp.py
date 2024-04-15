import subprocess
import numpy as np

def generate_data(n, m, l):
    with open('data', 'w') as file:
        file.write(f"{n} {m} {l}\n")
        matrix_a = np.random.randint(1, 100, size=(n, m))
        matrix_b = np.random.randint(1, 100, size=(m, l))

        np.savetxt(file, matrix_a, fmt='%d')

        np.savetxt(file, matrix_b, fmt='%d')

def main():
    generate_data(5, 4, 3)  # 修改矩陣的維度

    subprocess.run(['mpirun', '-np', '4', '--hostfile', 'hosts', '/home/312551129/matmul', '<', 'data'])

    # 讀取計算結果並儲存到 output.txt
    with open('result.txt', 'r') as result_file, open('output.txt', 'w') as output_file:
        for line in result_file:
            output_file.write(line)

    # 計算 a*b 的結果並儲存到 expected.txt
    matrix_a = np.loadtxt('data', skiprows=1, max_rows=5)
    matrix_b = np.loadtxt('data', skiprows=7, max_rows=4)
    result_ab = np.dot(matrix_a, matrix_b)
    np.savetxt('expected.txt', result_ab, fmt='%d')

    # 比較計算結果與 a*b 的結果
    result_is_correct = np.array_equal(np.loadtxt('output.txt', dtype=int), np.loadtxt('expected.txt', dtype=int))
    if result_is_correct:
        print("計算結果正確！")
    else:
        print("計算結果不正確！")

if __name__ == "__main__":
    main()
