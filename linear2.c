/**
 * @file nn.c
 * @brief 利用神经网络权重衰减确定线性方程 $y=\Sigma_{n=0}^p k_nx^n + b_n$ 的全部参数
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
#define DATA_SIZE 150
#define TRAIN_SIZE 110
#define VAL_SIZE 20
#define TEST_SIZE 20

// 训练参数
#define EPOCHS 2500         // 训练次数
#define LEARNING_RATE 0.005 // 学习率

// 多项式参数
#define TIMES 3 // 多项式次数

typedef struct
{
    double x;
    double y;
} Point;

typedef struct
{
    double k;
    double b;
} Item;

typedef struct
{
    int size;
    Item *items;
} Polynomial;

/**
 * @brief 初始化多项式参数
 * @param poly 多项式参数
 */
void init(Polynomial *poly)
{
    poly->size = TIMES + 1;
    poly->items = (Item *)malloc(poly->size * sizeof(Item));
    for (int i = 0; i < poly->size; i++)
    {
        poly->items[i].k = (double)rand() / RAND_MAX;
        poly->items[i].b = (double)rand() / RAND_MAX;
    }
}

/**
 * @brief 多项式模型
 * @param poly 多项式参数
 * @param x 输入
 * @return 输出
 */
double predict(Polynomial *poly, double x)
{
    double y = 0;
    for (int i = 0; i < poly->size; i++)
    {
        y += poly->items[i].k * pow(x, i) + poly->items[i].b;
    }
    return y;
}

/**
 * @brief 损失函数
 * @param poly 多项式参数
 * @param data 数据集
 * @param size 数据集大小
 * @return 损失
 */
double loss(Polynomial *poly, Point *data, int size)
{
    double sum = 0;
    for (int i = 0; i < size; i++)
    {
        double y_pred = predict(poly, data[i].x);
        sum += (y_pred - data[i].y) * (y_pred - data[i].y);
    }
    return sum / size;
}

/**
 * @brief 梯度下降
 * @param poly 多项式参数
 * @param data 数据集
 * @param size 数据集大小
 * @param learning_rate 学习率
 */
void gradient_descent(Polynomial *poly, Point *data, int size, double learning_rate)
{
    for (int i = 0; i < size; i++)
    {
        double y_pred = predict(poly, data[i].x);
        for (int j = 0; j < poly->size; j++)
        {
            poly->items[j].k -= learning_rate * (y_pred - data[i].y) * pow(data[i].x, j);
            poly->items[j].b -= learning_rate * (y_pred - data[i].y);
        }
    }
}

/**
 * @brief 训练
 * @param poly 多项式参数
 * @param train 数据集
 * @param train_size 数据集大小
 * @param val 数据集
 * @param val_size 数据集大小
 * @param epochs 训练次数
 * @param learning_rate 学习率
 */
void train(Polynomial *poly, Point *train, int train_size, Point *val, int val_size, int epochs, double learning_rate)
{
    for (int i = 0; i < epochs; i++)
    {
        gradient_descent(poly, train, train_size, learning_rate);
        if (i % 100 == 0)
        {
            printf("Epoch %d, Train Loss: %lf, Val Loss: %lf\n", i, loss(poly, train, train_size), loss(poly, val, val_size));
        }
    }
}

/**
 * @brief 测试
 * @param poly 多项式参数
 * @param test 数据集
 * @param test_size 数据集大小
 */
void test(Polynomial *poly, Point *test, int test_size)
{
    double loss = 0;
    for (int i = 0; i < test_size; i++)
    {
        double y_pred = predict(poly, test[i].x);
        loss += (y_pred - test[i].y) * (y_pred - test[i].y);
        printf("x: %lf, y: %lf, y_pred: %lf\n", test[i].x, test[i].y, y_pred);
    }
    printf("Test Loss: %lf\n", loss / test_size);
}

/**
 * Read data from file
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
    Point data[DATA_SIZE];
    read_data(DATA_FILE, data);
    shuffle(data, DATA_SIZE);
    Point *train_data = data;
    Point *val_data = data + TRAIN_SIZE;
    Point *test_data = data + TRAIN_SIZE + VAL_SIZE;

    Polynomial poly;
    init(&poly);

    train(&poly, train_data, TRAIN_SIZE, val_data, VAL_SIZE, EPOCHS, LEARNING_RATE);
    test(&poly, test_data, TEST_SIZE);

    return 0;
}
