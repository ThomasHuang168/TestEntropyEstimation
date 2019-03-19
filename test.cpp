#include<iostream>
#include<opencv2/opencv.hpp>
#include<fstream>
#include<vector>
#include<numeric>

using namespace std;
using namespace cv;

double variance(vector <double> v) {
	double sum = std::accumulate(std::begin(v), std::end(v), 0.0);
	double mean = sum / v.size();

	double accum = 0.0;
	std::for_each(std::begin(v), std::end(v), [&](const double d) {
		accum += (d - mean)*(d - mean);
	});

	double var = accum / (v.size());
	return var;
}

size_t combination(size_t n, size_t r)
{
	if (r > n)
	{
		return 0;
	}
	size_t numerator = 1, denomiator = 1;
	for (size_t i = 0; i < r; i++)
	{
		denomiator *= (i + 1);
		numerator *= n - i;
	}
	return numerator / denomiator;
}

class EntropyEstimation
{
public:
	//Compute Number of computation using order specified
	size_t GetNumberofEntropyComputation(const size_t order) const
	{
		//const size_t order = 4;
		if (order < 0 || order >= m_numElement)
		{
			cerr << "order (" << order << ") is out of range [0, " << m_numElement << ")" << endl;
			return 0;
		}
		else
		{
			cout << "For each Cross Validation Fold, There are:" << endl;
			size_t numComputation = m_numElement;
			cout << m_numElement << " variance computations" << endl;
			for (size_t i = 2; i <= order; i++)
			{
				size_t numComput = i * combination(m_numElement, i);
				cout << numComput << " " << i << "-conditional entropy computations" << endl;
				numComputation += numComput;
			}
			cout << numComputation << " Total (entropy|variance) computations" << endl;
			return numComputation;
		}
	}

	size_t Index(size_t number, vector<size_t> cond) const
	{
		size_t index = -1, indexBase = 0;
		size_t condSize = cond.size();
		if (condSize == 0)
		{
			index = number * m_numElement;
		}
		sort(cond.begin(), cond.end());
		for (size_t i = condSize - 1; i >= 0; i--)
		{
			indexBase +=  combination(m_numElement - 1, i);
		}
		size_t prev = 0;
		size_t subsetSize = condSize;
		size_t i = cond.size() - 1;
		for (size_t j = m_numElement - 1; j >= 0; j--)
		{
			if (cond[i] == j)
			{
				if (i > j)
				{
					cout << "cond[" << i << "] = " << cond[i] << " == j (" << j << ")" << endl;
					indexBase += combination(cond[i], i + 1);
				}
			}
		}
		index = indexBase;
		return index;
	}

	//CombinationIndex iterator element in combination increasing order
	//zero-based comb 
	//order is number of element in universe
	//for example
	//	Index	|Elements (order = 4)
	//	0		|0000
	//	1		|0001
	//	2		|0010
	//	3		|0100
	//	4		|1000
	//	5		|0011
	//	6		|0101
	//	7		|0110
	//	8		|1001
	//	9		|1010
	//	10		|1100
	//	11		|0111
	//	12		|1011
	//	13		|1101
	//	14		|1110
	//	15		|1111
	static size_t combinationIndex(size_t order, vector<size_t> comb, bool verbose = false)
	{
		if (verbose)
		{
			cout << "Find index of {";
			for (int i = comb.size() - 1; i >= 0; i--)
			{
				cout << comb[i] << ", ";
			}
			cout << "} in order " << order << endl;
		}

		size_t index = 0;
		size_t condSize = comb.size();
		sort(comb.begin(), comb.end());
		if (verbose)
		{
			cout << "it is ";
		}
		//for (int i = condSize - 1; i >= 0; i--)
		for (int i = 0; i < condSize; i++)
		{
			index += combination(order, i);
			if (verbose)
			{
				cout << order << "C" << i << "  ";
			}
		}
		size_t prev = 0;
		size_t subsetSize = condSize;
		int  i = comb.size() - 1;
		for (int i = condSize - 1; i >= 0; i--)
		{
			if (comb[i] > i)
			{
				index += combination(comb[i], i + 1);
				if (verbose)
				{
					cout << comb[i] << "C" << i + 1 << "  ";
				}
			}
		}
		if (verbose)
		{
			cout << endl;
			cout << index << endl;
		}
		return index;
	}

	static void TestCombinationIndexAndVector(size_t order)
	{
		size_t error = 0;
		size_t pow = 1 << order;
		cout << "\nTest Comb Index 2 Vector with order " << order << endl;
		for (size_t i = 0; i < pow; i++)
		{
			cout << i << "\t";
			vector<size_t> comb;
			comb = EntropyEstimation::combinationIndex2Vector(order, i);

			size_t result = EntropyEstimation::combinationIndex(order, comb);
			cout << result << "\t";

			if (i != result)
			{
				error++;
			}

			sort(comb.begin(), comb.end());
			for (int j = order - 1, k = comb.size() - 1; j >= 0; j--)
			{
				if (k < 0)
				{
					cout << "0";
				}
				else if (j == comb[k])
				{
					cout << "1";
					k--;
				}
				else
				{
					cout << "0";
				}
			}
			cout << endl;
		}
		cout << "error: " << error << endl;
	}

	static void TestCombinationIndex(size_t order)
	{
		size_t pow = 1 << order;
		for (size_t i = 0; i < pow; i++)
		{
			size_t curComb = i;
			vector<size_t> comb;
			for (size_t div = pow / 2, j = order - 1; div > 0; div /= 2, j--)
			{
				if (curComb / div == 1)
				{
					comb.push_back(j);
					curComb %= div;
				}
			}
			EntropyEstimation::combinationIndex(order, comb, true);
		}
	}

	static void TestCombinationIndex2Vector(size_t order)
	{
		size_t pow = 1 << order;
		cout << "\nTest Comb Index 2 Vector with order " << order << endl;
		for (size_t i = 0; i < pow; i++)
		{
			cout << i << "\t";
			vector<size_t> comb;
			comb = EntropyEstimation::combinationIndex2Vector(order, i);
			sort(comb.begin(), comb.end());
			for (int j = order - 1, k = comb.size() - 1; j >= 0; j--)
			{
				if (k < 0)
				{
					cout << "0";
				}
				else if (j == comb[k])
				{
					cout << "1";
					k--;
				}
				else
				{
					cout << "0";
				}
			}
			cout << "\n";
		}
	}

	static vector<size_t> combinationIndex2Vector(size_t order, size_t index)
	{
		vector<size_t> Output;
		//Get number of Element by order defined in 
		//CombinationIndex
		//use decomposition of combination
		//if index >= (order)C0, that means there are at least one element
		//in the vector
		size_t numOutput = 0, tempSum = 0;
		for (numOutput = 0; numOutput < order; numOutput++)
		{
			size_t comb = combination(order, numOutput);
			if (index >= tempSum + comb)
			{
				tempSum += comb;
			}
			else
			{
				break;
			}
		}
		if (numOutput >= order && (index > tempSum + combination(order, numOutput)))
		{
			cerr << "Unexpected result" << endl;
		}
		else
		{
			//Get Exact Elements
			//use decomposition of combination by order defined in 
			//CombinationIndex
			//i.e. 4C2 = 1C0*3C2 + 1C1*3C1
			//First find exactance of most weighted element (3 in ab. ex)
			//if (index - tempSum) >= 3C2, that means 3 is in the vector
			int element;
			for (element = order - 1; element >= 0; element--)
			{
				if (index - tempSum == 0)
				{
					//Fill in remaining with the smallest elements
					for (size_t i = 0; i < numOutput; i++)
					{ 
						Output.push_back(i);
					}
					break;
				}
				else if (element >= numOutput)
				{
					if (0 == numOutput)
					{
						break;
					}
					if ((index - tempSum) >= combination(element, numOutput))
					{
						tempSum += combination(element, numOutput);
						Output.push_back(element);
						numOutput--;
					}
				}
				else
				{
					cerr << "Unexpected result" << endl;
					break;
				}
			}
		}
		return Output;
	}
	
	//encode element in EntropyTable
	//zero-based vect and indResp
	//return zero-based Table index
	size_t getIndex(vector<size_t> vect, size_t indResp)
	{
		vector<size_t>::iterator it;
		it = find(vect.begin(), vect.end(), indResp);
		if (it != vect.end())
		{
			cerr << "found vect element same as indResp" << endl;
			return -1;
		}
		else if (vect.size() >= m_numElement)
		{
			cerr << "found vect size same as num of elements" << endl;
			return -1;
		}
		sort(vect.begin(), vect.end());
		vector<size_t> eleByResp;
		for (auto &ele : vect)
		{
			if (ele > indResp)
			{
				ele--;
			}
			eleByResp.push_back(ele);
		}
		size_t indexInResp = combinationIndex(m_numElement, eleByResp);
		size_t index = indResp * m_tableCols + indexInResp;
		return index;
	}

	//Reverse of getIndex
	vector<size_t> getElement(size_t index, size_t indResp)
	{
		vector<size_t> vect = combinationIndex2Vector(m_numElement - 1, index);
		vector<size_t> result;
		for (auto &ele : vect)
		{
			if (ele >= indResp)
			{
				ele++;
			}
			result.push_back(ele);
		}
		return result;
	}

	size_t buildTable(string filename, size_t order)
	{
		m_order = order;
		//LoadFromCSV
		Ptr<ml::TrainData> PtrData;
		PtrData = ml::TrainData::loadFromCSV(filename.c_str(), 0);
		Mat MatSample, MatResp, MatData;
		MatSample = PtrData->getTrainSamples();
		MatResp = PtrData->getTrainResponses();
		hconcat(MatSample, MatResp, MatData);
		MatSample.release();
		MatResp.release();
		cout << "MatData:\n" << MatData << endl;

		//Split into n Datasets
		nCVFold = 3;
		m_numSample = MatData.size().height;
		m_numElement = MatData.size().width;
		cout << m_numSample << " - " << m_numElement << endl;
		if (m_numSample < nCVFold)
		{
			cerr << "height (" << m_numSample << ") is less than nCVFold (" << nCVFold << ")" << endl;
			return 1;
		}
		size_t stepSize = m_numSample / nCVFold;
		if (0 != m_numSample % nCVFold)
		{
			stepSize++;
		}

		int TableSize = GetNumberofEntropyComputation(order);
		EntropyTable.resize(TableSize);
		if (TableSize != EntropyTable.size())
		{
			cerr << TableSize << " != " << EntropyTable.size();
			return 1;
		}
		else
		{
			cout << "There are " << TableSize << "of var/entropy computations, \n which is (" << m_numElement << " * " << TableSize / m_numElement << " + " << TableSize % m_numElement << ")" << endl;
		}

		//Compute all variance and conditional entropy 
		for (size_t curStep = 0; curStep < m_numSample; curStep += stepSize)
		{
			//Generate Train and Test;
			Mat MatTest = MatData(Range(curStep, min(curStep + stepSize, m_numSample)), Range::all());
			//cout << "MatTest:\n" << MatTest << endl;
			Mat MatTrain;
			if (curStep == 0)
			{
				MatTrain = MatData(Range(curStep + stepSize, m_numSample), Range::all());
			}
			else if (curStep + stepSize >= m_numSample)
			{
				MatTrain = MatData(Range(0, curStep), Range::all());
			}
			else
			{
				vconcat(MatData(Range(0, curStep), Range::all()), MatData(Range(curStep + stepSize, m_numSample), Range::all()), MatTrain);
			}
			//cout << "MatTrain:\n" << MatTrain << endl;

			//Select Responce
			size_t indResp = 0;
			//iterate all train dataset combinations
			Mat MatTrainSample, MatTrainResp, MatTestSample, MatTestResp;
			size_t combinationPerResp = 0;
			for (int i = 0; i < order; i++)
			{
				combinationPerResp += combination(m_numElement - 1, i);
			}
			m_tableCols = combinationPerResp;
			{
				//compute variance (index = 0)
				float TestVariance;
				MatTrainResp = MatTrain(Range::all(), Range(indResp, indResp + 1));
				cout << "MatTrainResp: \n" << MatTrainResp << endl;
				MatTestResp = MatTest(Range::all(), Range(indResp, indResp + 1));
				cout << "MatTestResp: \n" << MatTestResp << endl;

				Scalar m, stdv;
				meanStdDev(MatTestResp, m, stdv);
				cout << m << stdv << endl;
				TestVariance = stdv[0] * stdv[0];
				EntropyTable[indResp * m_tableCols] = TestVariance;
			}
			for (int index = 1; index < combinationPerResp; index++)
			{
				if (indResp < 0 || indResp >= m_numElement)
				{
					cerr << "Index of Responce (" << indResp << ") is out of Range [0, " << m_numElement << ")" << endl;
					return 1;
				}

				//Select Sample set by index
				vector<size_t> sampleSet = getElement(index, indResp);
				if (sampleSet.size() < 1)
				{
					cerr << "unexpected result" << endl;
					return 1;
				}
				else 
				{
					MatTrainSample = MatTrain(Range::all(), Range(sampleSet[0], sampleSet[0] + 1));
					MatTestSample = MatTest(Range::all(), Range(sampleSet[0], sampleSet[0] + 1));
					for (int i = 1; i < sampleSet.size(); i++)
					{
						hconcat(MatTrainSample, MatTrain(Range::all(), Range(sampleSet[i], sampleSet[i] + 1)), MatTrainSample);
						hconcat(MatTestSample, MatTest(Range::all(), Range(sampleSet[i], sampleSet[i] + 1)), MatTestSample);
					}
				}
				//if (indResp == 0)
				//{
				//	MatTrainSample = MatTrain(Range::all(), Range(indResp + 1, m_numElement));
				//	MatTestSample = MatTest(Range::all(), Range(indResp + 1, m_numElement));
				//}
				//else if (indResp == m_numElement - 1)
				//{
				//	MatTrainSample = MatTrain(Range::all(), Range(0, indResp));
				//	MatTestSample = MatTest(Range::all(), Range(0, indResp));
				//}
				//else
				//{
				//	hconcat(MatTrain(Range::all(), Range(0, indResp)), MatTrain(Range::all(), Range(indResp + 1, m_numElement)), MatTrainSample);
				//	hconcat(MatTest(Range::all(), Range(0, indResp)), MatTest(Range::all(), Range(indResp + 1, m_numElement)), MatTestSample);
				//}
				cout << "MatTrainSample: \n" << MatTrainSample << endl;
				cout << "MatTestSample: \n" << MatTestSample << endl;

				Ptr<ml::TrainData> PtrTrain = ml::TrainData::create(MatTrainSample, ml::ROW_SAMPLE, MatTrainResp);
				Ptr<ml::TrainData> PtrTest = ml::TrainData::create(MatTestSample, ml::ROW_SAMPLE, MatTestResp);

				//Create a Model
				Ptr<ml::DTrees> dtree = ml::DTrees::create();

				//Set parameters
				dtree->setMaxDepth(5);
				dtree->setMinSampleCount(10);
				dtree->setRegressionAccuracy(0.01f);
				dtree->setUseSurrogates(false /* true */);
				dtree->setMaxCategories(15);
				dtree->setCVFolds(0 /*10*/); // nonzero causes core dump
				dtree->setUse1SERule(true);
				dtree->setTruncatePrunedTree(true);
				dtree->setPriors(cv::Mat()); // ignore priors for now...

				dtree->train(PtrTrain);

				Mat results;

				float performance = dtree->calcError(PtrTest, false, results);
				cout << "performance: " << performance << endl;
				vector<size_t> cond;
				EntropyTable[index + indResp * combinationPerResp] = performance;
			}
		}
		return 0;
	}
	size_t m_tableRows;
	size_t m_tableCols, m_numElement, m_order, m_numSample, nCVFold;
	vector< float> EntropyTable;
	vector<size_t> orderIndex;
};

int main(int argc, char **argv)
{
	std::ifstream f;
	string model;
	if (argc > 2)
	{
		f.open(argv[1]);
		if (!f.is_open())
		{
			std::cerr << "error: file open failed '" << argv[1] << "'.\n";
			return 1;
		}
		model = argv[2];
	}
	else
	{
		std::cerr << "error: insufficient input. <filename> <model(\"cart\" or \"svr\")required.\n";
		return 1;
	}

	//vector<vector<double>> array;
	//{
	//	string line, val;
	//	while (getline(f, line))
	//	{
	//		vector<double> v;
	//		stringstream s(line);
	//		while (getline(s, val, ','))
	//		{
	//			v.push_back(std::stod(val));
	//		}
	//		array.push_back(v);
	//	}

	//	for (auto& row : array)
	//	{
	//		for (auto& val : row)
	//		{
	//			cout << val << "\t";
	//		}
	//		cout << "\n";
	//	}
	//	fflush(stdout);
	//}

	//Ptr<ml::TrainData> PtrTrainData, PtrTestData;
	//{
	//	vector<vector<double>> TrainArray, TestArray;
	//	int i = 0;
	//	for (auto& subArray : array)
	//	{
	//		if (i % 2)
	//		{
	//			TrainArray.push_back(subArray);
	//		}
	//		else
	//		{
	//			TestArray.push_back(subArray);
	//		}
	//	}
	//	PtrTrainData = ml::TrainData::create(TrainArray, ml::ROW_SAMPLE, noArray());
	//	PtrTestData = ml::TrainData::create(TestArray, ml::ROW_SAMPLE, noArray());
	//}
	////for each dataset
	////treat it as test dataset 
	////treat other as train dataset
	//average out the calerror by cvFolds
	//write the error into output table
	//Python wrapper

	EntropyEstimation EE;
	EE.buildTable(argv[1], 2);

	//size_t n = 5;
	//size_t pow = 2 * 2 * 2 * 2 * 2;
	//for (size_t i = 0; i < pow; i++)
	//{
	//	size_t temp = i;
	//	cout << endl;
	//}



	return 0;
}