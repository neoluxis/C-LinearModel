#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define NUM_DATA_POINTS 500
#define NUM_TRAINING_SAMPLES 350
#define NUM_VALIDATION_SAMPLES 100
#define NUM_TEST_SAMPLES 50

#define LEARNING_RATE 0.001
#define EPOCHS 1000

// 数据点结构体
typedef struct
{
    double x;
    double y;
} DataPoint;

/**
 * 线性模型结构体
 * w1, w2, w3, w4: 模型参数 权重
 * b1, b2, b3, b4: 模型参数 偏置
 */
typedef struct
{
    double w1;
    double b1;
    double w2;
    double b2;
    double w3;
    double b3;
    double w4;
    double b4;
} LinearModel;

/**
 * 读取数据
 * @param data_points 数据点数组
 * @param file_name 文件名
 */
void read_data(DataPoint *data_points, const char *file_name)
{
    FILE *file = fopen(file_name, "r");
    if (file == NULL)
    {
        fprintf(stderr, "Error opening file.\n");
        exit(1);
    }

    for (int i = 0; i < NUM_DATA_POINTS; ++i)
    {
        fscanf(file, "%lf,%lf\n", &data_points[i].x, &data_points[i].y);
    }

    fclose(file);
}

/**
 * 打乱数组元素
 * @param arr 待打乱的数组
 * @param len 数组长度
 */
void shuffle(int *arr, int len)
{
    // 使用当前时间作为随机种子
    srand(time(NULL));

    // 从最后一个元素开始，逐个与随机位置的元素交换
    for (int i = len - 1; i > 0; i--)
    {
        // 生成一个随机位置
        int j = rand() % (i + 1);

        // 交换元素
        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }
}

/**
 * 初始化线性模型
 * 为了简化，这里将所有参数初始化为0, 实际中可以使用随机值初始化或者正太分布初始化
 * @param model 线性模型
 */
void initialize_linear_model(LinearModel *model)
{
    model->w1 = 0.0;
    model->b1 = 0.0;
    model->w2 = 0.0;
    model->b2 = 0.0;
    model->w3 = 0.0;
    model->b3 = 0.0;
    model->w4 = 0.0;
    model->b4 = 0.0;
}

// 计算模型预测值
double predict(const LinearModel *model, double x)
{
    x = model->w1 * x + model->b1;
    x = model->w2 * x + model->b2;
    x = model->w3 * x + model->b3;
    x = model->w4 * x + model->b4;
    return x; // y_hat = w4 * (w3 * (w2 * (w1 * x + b1) + b2) + b3) + b4
}

// 训练模型
void train_model(LinearModel *model, DataPoint *training_set)
{
}

int main()
{

    return 0;
}