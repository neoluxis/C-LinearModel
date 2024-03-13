/**
 * @file nn.c
 * @brief 利用神经网络权重衰减确定线性回归 y = kx + b 的参数 k 和 b
 * @version 0.1
 * @date 2024-03-01
 * @author Neolux Lee
 */
#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "time.h"

// 数据参数
#define INPUT_SIZE 1
#define OUTPUT_SIZE 1
#define DATA_FILE "data.txt"

// 数据集划分
#define DATA_SIZE 5000
#define TRAIN_SIZE 4000
#define VAL_SIZE 500
#define TEST_SIZE 500

// 训练参数
#define EPOCHS 300        // 训练次数
#define LEARNING_RATE 0.01 // 学习率

typedef struct
{
    double x;
    double y;
} Point;

typedef struct
{
    double w;
    double b;
} Linear;

/**
 * @brief 初始化线性回归参数
 * @param linear 线性回归参数
 */
void init_linear(Linear *linear)
{
    linear->w = (double)rand() / RAND_MAX;
    linear->b = (double)rand() / RAND_MAX;
}

/**
 * @brief 线性回归模型
 * @param linear 线性回归参数
 * @param x 输入
 * @return 输出
 */
double predict(Linear *linear, double x)
{
    return linear->w * x + linear->b;
}

/**
 * @brief 损失函数 Cross Entropy
 * @param linear 线性回归参数
 * @param data 数据集
 * @param size 数据集大小
 * @return 损失
 */
double loss(Linear *linear, Point *data, int size)
{
    double sum = 0;
    for (int i = 0; i < size; i++)
    {
        double y_pred = predict(linear, data[i].x);
        sum += (y_pred - data[i].y) * (y_pred - data[i].y);
    }
    return sum / size;
}

/**
 * @brief 梯度下降
 * @param linear 线性回归参数
 * @param data 数据集
 * @param size 数据集大小
 * @param learning_rate 学习率
 */
void gradient_descent(Linear *linear, Point *data, int size, double learning_rate)
{
    double dw = 0;
    double db = 0;
    for (int i = 0; i < size; i++)
    {
        double y_pred = predict(linear, data[i].x);
        dw += (y_pred - data[i].y) * data[i].x;
        db += (y_pred - data[i].y);
    }
    dw /= size;
    db /= size;
    linear->w -= learning_rate * dw;
    linear->b -= learning_rate * db;
}

/**
 * @brief 训练模型
 * @param linear 线性回归参数
 * @param train 数据集
 * @param train_size 数据集大小
 * @param val 数据集
 * @param val_size 数据集大小
 * @param epochs 训练次数
 * @param learning_rate 学习率
 */
void train(Linear *linear, Point *train, int train_size, Point *val, int val_size, int epochs, double learning_rate)
{
    for (int i = 0; i < epochs; i++)
    {
        gradient_descent(linear, train, train_size, learning_rate);
        printf("Epoch: %d, Loss: %f\n", i, loss(linear, val, val_size));
    }
}

/**
 * @brief 测试模型
 * @param linear 线性回归参数
 * @param data 数据集
 * @param size 数据集大小
 */
void test(Linear *linear, Point *data, int size)
{
    double loss = 0;
    for (int i = 0; i < size; i++)
    {
        double y_pred = predict(linear, data[i].x);
        loss += (y_pred - data[i].y) * (y_pred - data[i].y);
        printf("x: %f, y: %f, y_pred: %f\n", data[i].x, data[i].y, y_pred);
    }
    printf("Loss: %f\n", loss / size);
}

/**
 * @brief 读取数据
 * Data format:
 * x1, y1
 * x2, y2
 * ...
 * @param filename File name
 * @param data Data array
 */
void read_data(const char *filename, Point *data)
{
    FILE *file = fopen(filename, "r");
    if (file == NULL)
    {
        printf("File not found\n");
        return;
    }
    int i = 0;
    while (fscanf(file, "%lf, %lf", &data[i].x, &data[i].y) != EOF)
    {
        i++;
    }
    fclose(file);
}

/**
 * @brief 打乱数组
 * @param arr 数组
 * @param len 数组长度
 */
void shuffle(Point *arr, int len)
{
    // 使用当前时间作为随机种子
    srand(time(NULL));

    // 从最后一个元素开始，逐个与随机位置的元素交换
    for (int i = len - 1; i > 0; i--)
    {
        // 生成一个随机位置
        int j = rand() % (i + 1);

        // 交换元素
        Point temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }
}

int main(int argc, char const *argv[])
{
    // 读取数据
    srand(time(NULL));
    Point data[DATA_SIZE];
    read_data(DATA_FILE, data);
    shuffle(data, DATA_SIZE);
    Point *train_data = data;
    Point *val_data = data + TRAIN_SIZE;
    Point *test_data = data + TRAIN_SIZE + VAL_SIZE;

    // 初始化线性回归参数
    Linear linear;
    init_linear(&linear);

    // 训练模型
    printf("Training...\n");
    train(&linear, train_data, TRAIN_SIZE, val_data, VAL_SIZE, EPOCHS, LEARNING_RATE);
    printf("Training finished\n");

    // 测试模型
    printf("Testing...\n");
    test(&linear, test_data, TEST_SIZE);
    printf("Testing finished\n");

    // 输出参数
    printf("y = %fx + %f\n", linear.w, linear.b);
}
