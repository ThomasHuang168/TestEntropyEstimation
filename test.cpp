#include<iostream>
#include<opencv2/opencv.hpp>
#include<fstream>
#include<vector>
#include<numeric>
#include <random>

using namespace std;
using namespace cv;
void writeCSV(string filename, Mat m)
{
	ofstream myfile;
	myfile.open(filename.c_str());
	myfile << cv::format(m, cv::Formatter::FMT_CSV) << std::endl;
	myfile.close();
}

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

	EntropyEstimation()
	{
		m_numCVFold = 3;
		m_order = 2;
	}

	EntropyEstimation(size_t CVFold, size_t order)
	{
		m_numCVFold = CVFold;
		m_order = order;
	}

	~EntropyEstimation()
	{

	}

	size_t GetOrder(void) const
	{
		return m_order;
	}

	void SetOrder(const size_t order)
	{
		m_order = order;
	}

	size_t GetCVFold(void) const
	{
		return m_numCVFold;
	}

	size_t SetCVFold(const size_t numCVFold)
	{
		m_numCVFold = numCVFold;
	}

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
	//indResp: index of Responce Column/Element
	//vect: index(es) of those Col/Element conditioning the Responce
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
		size_t index = indResp * m_tableRows + indexInResp;
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

	size_t buildTable(string filename, string model, bool verbose = false) 
	{
		//LoadFromCSV
		Ptr<ml::TrainData> PtrData;
		PtrData = ml::TrainData::loadFromCSV(filename.c_str(), 0);
		Mat MatSample, MatResp, MatData;
		MatSample = PtrData->getTrainSamples();
		MatResp = PtrData->getTrainResponses();
		hconcat(MatSample, MatResp, MatData);
		MatSample.release();
		MatResp.release();
		if (verbose)
		{
			cout << "MatData:\n" << MatData << endl;
		}

		m_numSample = MatData.size().height;
		m_numElement = MatData.size().width;
		if (verbose)
		{
			cout << m_numSample << " - " << m_numElement << endl;
		}

		int TableSize = GetNumberofEntropyComputation(m_order);
		EntropyTable.resize(TableSize);
		if (TableSize != EntropyTable.size())
		{
			cerr << TableSize << " != " << EntropyTable.size();
			return 1;
		}
		else if (verbose)
		{
			cout << "There are " << TableSize << " of var/entropy computations, \n which is (" << m_numElement << " (num of RV) * " << TableSize / m_numElement << " (num of sample/RV) + " << TableSize % m_numElement << ")" << endl;
		}

		size_t combinationPerResp = 0;
		for (int i = 0; i < m_order; i++)
		{
			combinationPerResp += combination(m_numElement - 1, i);
		}
		m_tableRows = combinationPerResp;

		//Compute all variance and conditional entropy 

		//Split into n Datasets
		if (m_numSample < m_numCVFold)
		{
			cerr << "height (" << m_numSample << ") is less than nCVFold (" << m_numCVFold << ")" << endl;
			return 1;
		}
		size_t stepSize = m_numSample / m_numCVFold;
		if (0 != m_numSample % m_numCVFold)
		{
			stepSize++;
		}

		//Select Responce
		size_t indResp = 0;
		for (;indResp < m_numElement; indResp++)
		{
			//cout << "MatTrain:\n" << MatTrain << endl;
			//iterate all train dataset combinations
			Mat MatTrainSample, MatTrainResp, MatTestSample, MatTestResp;
			//new variable to replace 
			Mat MatResp, MatSample;
			{
				MatResp = MatData(Range::all(), Range(indResp, indResp + 1));
			}
			for (int index = 0; index < combinationPerResp; index++)
			{
				if (indResp < 0 || indResp >= m_numElement)
				{
					cerr << "Index of Responce (" << indResp << ") is out of Range [0, " << m_numElement << ")" << endl;
					return 1;
				}
				if (0 == index)
				{
					//compute variance (index = 0)
					float TestVariance = 0;
					if (verbose)
					{
						cout << "MatResp: \n" << MatResp << endl;
					}

					Scalar m, stdv;
					for (size_t curStep = 0; curStep < m_numSample; curStep += stepSize)
					{
						/********************************************************************************/
						MatTrainResp.release();
						MatTestResp.release();
						if (curStep == 0)
						{
							MatTrainResp = MatResp(Range(curStep + stepSize, m_numSample), Range::all());
						}
						else if (curStep + stepSize >= m_numSample)
						{
							MatTrainResp = MatResp(Range(0, curStep), Range::all());
						}
						else
						{
							vconcat(MatResp(Range(0, curStep), Range::all()), MatResp(Range(curStep + stepSize, m_numSample), Range::all()), MatTrainResp);
						}
						if (verbose)
						{
							cout << "MatTrainResp:\n" << MatTrainResp << endl;
						}

						MatTestResp = MatResp(Range(curStep, min(curStep + stepSize, m_numSample)), Range::all());
						if (verbose)
						{
							cout << "MatTestResp:\n" << MatTestResp << endl;
						}
						/********************************************************************************/

						meanStdDev(MatTestResp, m, stdv);
						cout << m << stdv << endl;
						TestVariance += stdv[0] * stdv[0];
					}
					EntropyTable[indResp * m_tableRows] = TestVariance / m_numCVFold;
				}
				else
				{
					//Select Sample set by index
					vector<size_t> sampleSet = getElement(index, indResp);
					if (sampleSet.size() < 1)
					{
						cerr << "unexpected result" << endl;
						return 1;
					}
					else
					{
						MatSample = MatData(Range::all(), Range(sampleSet[0], sampleSet[0] + 1));
						for (int i = 1; i < sampleSet.size(); i++)
						{
							hconcat(MatSample, MatData(Range::all(), Range(sampleSet[i], sampleSet[i] + 1)), MatSample);
						}
					}
					if (verbose)
					{
						cout << "MatSample: \n" << MatSample << endl;
					}

					float sumPerf = 0.0;
					for (size_t curStep = 0; curStep < m_numSample; curStep += stepSize)
					{
						/********************************************************************************/
						MatTrainResp.release();
						MatTestResp.release();
						if (curStep == 0)
						{
							MatTrainResp = MatResp(Range(curStep + stepSize, m_numSample), Range::all());
						}
						else if (curStep + stepSize >= m_numSample)
						{
							MatTrainResp = MatResp(Range(0, curStep), Range::all());
						}
						else
						{
							vconcat(MatResp(Range(0, curStep), Range::all()), MatResp(Range(curStep + stepSize, m_numSample), Range::all()), MatTrainResp);
						}
						if (verbose)
						{
							cout << "MatTrainResp:\n" << MatTrainResp << endl;
						}

						MatTestResp = MatResp(Range(curStep, min(curStep + stepSize, m_numSample)), Range::all());
						if (verbose)
						{
							cout << "MatTestResp:\n" << MatTestResp << endl;
						}

						MatTrainSample.release();
						MatTestSample.release();
						if (curStep == 0)
						{
							MatTrainSample = MatSample(Range(curStep + stepSize, m_numSample), Range::all());
						}
						else if (curStep + stepSize >= m_numSample)
						{
							MatTrainSample = MatSample(Range(0, curStep), Range::all());
						}
						else
						{
							vconcat(MatSample(Range(0, curStep), Range::all()), MatSample(Range(curStep + stepSize, m_numSample), Range::all()), MatTrainSample);
						}
						if (verbose)
						{
							cout << "MatTrainSample:\n" << MatTrainSample << endl;
						}

						MatTestSample = MatSample(Range(curStep, min(curStep + stepSize, m_numSample)), Range::all());
						if (verbose)
						{
							cout << "MatTestSample:\n" << MatTestSample << endl;
						}
						/********************************************************************************/

						Ptr<ml::TrainData> PtrTrain = ml::TrainData::create(MatTrainSample, ml::ROW_SAMPLE, MatTrainResp);
						Ptr<ml::TrainData> PtrTest = ml::TrainData::create(MatTestSample, ml::ROW_SAMPLE, MatTestResp);

						float performance = 0.0;

						Mat results;

						if ("cart" == model)
						{
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

							performance = dtree->calcError(PtrTest, false, results);
						}
						else if ("svr" == model)
						{
							Ptr<ml::SVM> svr = ml::SVM::create();
							svr->setKernel(ml::SVM::KernelTypes::LINEAR);
							svr->setGamma(10.0);
							svr->setDegree(0.1);
							svr->setCoef0(0.0);
							svr->setNu(0.1);
							svr->setP(0.1);
							svr->train(PtrTrain);
							if (!svr->isTrained())
							{
								cerr << "error occur during SVR training" << endl;
							}
							performance = svr->calcError(PtrTest, false, results);
						}
						else if ("svmsgd" == model)
						{
							Ptr<ml::SVMSGD> svmsgd = ml::SVMSGD::create();
							svmsgd->setSvmsgdType(ml::SVMSGD::ASGD);
							svmsgd->train(PtrTrain);
							if (!svmsgd->isTrained())
							{
								cerr << "error occur during SVR training" << endl;
							}
							performance = svmsgd->calcError(PtrTest, false, results);
						}
						if (verbose)
						{
							cout << "performance: " << performance << endl;
						}
						sumPerf += performance;
					}
					EntropyTable[index + indResp * combinationPerResp] = sumPerf / m_numCVFold;
				}
			}
		}
		return 0;
	}

	void writeTableCSV(string filename)
	{
		ofstream csv;
		csv.open(filename.c_str());
		if (csv.is_open())
		{
			size_t Cols = m_numElement, Rows = m_tableRows;
			if (EntropyTable.size() != Rows * Cols)
			{
				cerr << "Inconsistent Table Size" << endl;
				return;
			}
			for (int i = 0; i < Rows; i++)
			{
				for (int j = 0; j < Cols; j++)
				{
					if (j != 0)
					{
						csv << ", ";
					}
					csv << EntropyTable[i + j * Rows];
				}
				csv << endl;
			}
			csv.close();
		}
	}

	size_t m_tableRows, m_numElement, m_order, m_numSample, m_numCVFold;
	vector< float> EntropyTable;
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

	//EntropyEstimation EE(3, 4);
	//EE.buildTable(argv[1], argv[2]);
	//EE.writeTableCSV("..\\output.csv");
	const int nrolls = 10000;
	const int nstars = 95;
	default_random_engine generator;
	uniform_int_distribution<int> distribution(0, 9);

	int p[10] = {};

	for (int i = 0; i < nrolls; ++i) {
		int number = distribution(generator);
		++p[number];
	}

	std::cout << "uniform_int_distribution (0,9):" << std::endl;
	for (int i = 0; i < 10; ++i)
	{
		std::cout << i << ": " << std::string(p[i] * nstars / nrolls, '*') << std::endl;
	}

	return 0;
}