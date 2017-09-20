#include <thread>
#include <iostream>
#include <vector>
#include<random>
#include <chrono>
#include <functional>
#include <fstream>
using namespace std;
using namespace std::chrono;

void task(int n, int val)
{
	cout << "Thread: " << n << " Random Value: " << val << endl;
}

auto add = [](int x, int y) {return x + y; };

function<int(int, int)> add_function = [](int x, int y)
{
	return x + y;
};


void work()
{
	// Spin, spin spin!
	int n = 0;
	for (int i = 0; i < 1000000; i++)
		++n;
}



void monte_carlo_pi(unsigned int it)
{
	auto millis = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
	default_random_engine e(millis.count());

	// Create distribution.
	uniform_real_distribution<double> distribution(0.0, 1.0);

	// Keep track of num of points in circle.
	unsigned int in_circle = 0;
	// iterate

	for (unsigned int i = 0; i < it;i++)
	{
		// Generate random point.
		auto x = distribution(e);
		auto y = distribution(e);

		// Get length.
		auto length = sqrt((x*x) + (y*y));
		// check if in circle.
		if (length <= 1.0)
			in_circle++;
	}

	// calc pi.
	auto pi = (4 * 0 * in_circle) / static_cast<double>(it);
}


int main()
{
	//// Create seed using time. 
	//auto millis = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
	//default_random_engine e(static_cast<unsigned int>(millis.count()));

	//// Create 100 threads.
	//vector<thread> threads;
	//for (int i = 0; i < 100; i++)
	//{
	//	threads.push_back(thread(task, i, e()));
	//}

	//// Join all the threads.
	//for (auto &t : threads)
	//	t.join();


	/* ---- 1.4------ */

	//auto x = add_function(10, 12);
	//cout << "10 + 12 = " << x << endl;
	

	/* ------ 1.5 -------  */

	//thread t([] {cout << "Hello, I am lambda thread!" << endl; });
	//t.join();


	/*  ----- 1.6 ------- */
	
	//ofstream data("data.csv", ofstream::out);
	//for (int i = 0; i < 100; i++)
	//{
	//	// Get start time.
	//	auto start = system_clock::now();
	//	// execute thread.
	//	thread t(work);
	//	t.join();
	//	// Get end time.
	//	auto end = system_clock::now();
	//	auto duration = end - start;
	//	// Convert to ms.
	//	data << duration_cast<milliseconds>(duration).count() << endl;
	//}
	//data.close();

	/* ------- 1.7 ---------*/

	ofstream data("montecarlo.csv", ofstream::out);
	for (unsigned int numThreads = 0; numThreads <= 8; numThreads++)
	{
		auto totalThreads = static_cast<unsigned int>(pow(2.0, numThreads));
		cout << "Number of threads = " << totalThreads << endl;
		data << "NumThreads " << totalThreads;

		// run 100 execuations.
		for (unsigned int iters = 0; iters <100; iters++)
		{
			auto start = system_clock::now();
			vector<thread> threads;
			for (unsigned int n = 0; n < totalThreads; n++)
			{
				threads.push_back(thread(monte_carlo_pi,static_cast<unsigned int>(pow(2.0, 24.0 -numThreads))));
			}
			for (auto &t : threads)
				t.join();
			auto end = system_clock::now();
			auto total = end - start;
			data <<", "<< duration_cast <milliseconds >(total).count();
		}
		data << endl;
	}
	data.close();


	system("pause");
	return 0;
}
