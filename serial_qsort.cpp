#include<iostream>
using namespace std;

int basic_sort(int array[], int i, int j)
{
    int temp = array[i];
    while (i < j)
    {
        while (i < j && array[j] >= temp)
                j--;
        if (array[j] < temp)
        {
            array[i] = array[j]; 
        }
        while (i < j && array[i] <= temp)
        {
            i++;
        }
        if (array[i] > temp)
        {
            array[j] = array[i]; 
        }
    }
    array[i] = temp;
    //cout << i << endl;
    return i;
}

void serial_psort(int array[], int low,int high)
{
  
    if (low < high)
    {
        int temp2 = basic_sort(array, low, high);
        serial_psort(array, low, temp2 - 1);
        serial_psort(array, temp2 + 1, high);
    }
    return;
}

int main()
{
    int arr[] = { 3, 6, 2, 5, 4, 8, 10, 15, 18, 0, -4, 8 };
    int size = sizeof(arr) / sizeof(int);
    serial_psort(arr, 0, size-1);
    for (int i = 0; i < size; i++)
    {
        cout << arr[i]<<endl;
    }
}
