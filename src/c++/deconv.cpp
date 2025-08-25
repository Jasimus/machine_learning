#include <iostream>
#include <unordered_map>
#include <vector>

using namespace std;

void indices_x(unordered_map<int,int> &p, int nz, int pad, vector<vector<int>> &x);
void indices_k(unordered_map<int,int> &p, int h_in, int nz, int pad, vector<vector<int>> &k);
vector<vector<int>> deconv(unordered_map<int, int> &px, unordered_map<int, int> &pk, int h_in, int ksize, int s);
vector<vector<int>> deconv_mejor(vector<vector<int>> &x, vector<vector<int>> &k, int s, int p);

int main(int argc, char *argv[])
{
	vector<vector<int>> k = {{-1, 1, 0},
                            {-2, 3, 1},
                            {1, 2, -1}};
    
    vector<vector<int>> k_rot = {{-1, 2, 1},
                             {1, 3, -2},
	                         {0, 1, -1}};
    
	                                           
	vector<vector<int>> x = {{2, 3},{-2, 1}};
	vector<vector<int>> Y;
	
	int s = 2;
	int ksize = k.size();
	int h_in = x.size();
	int nz = s - 1, p = ksize - 1;
	
	// unordered_map <int, int> pos_x;
	// unordered_map <int, int> pos_k;
	
	// indices_x(pos_x, nz, p, x);
	// indices_k(pos_k, h_in, nz, p, k);
	// Y = deconv(pos_x, pos_k, h_in, ksize, s);
	
    Y = deconv_mejor(x, k, s, p);

	for (auto f:Y)
	{ 
	    for(auto e:f)
    	{
	        cout << e << ' ';
    	}
    	cout << '\n';
	}
}

void indices_x(unordered_map<int,int> &p, int nz, int pad, vector<vector<int>> &x)
{
    int h_in = x.size();
    int ind = (h_in + nz + 2*pad)*pad + pad;
    
    
    for (int i=0; i<h_in; i++)
    {
        for (int j=0; j<h_in; j++)
        {
            p[ind] = x[i][j];
            if(j != h_in - 1) ind += nz + 1 ;
        }
        ind += 2*pad + (h_in + nz + 2*pad) * nz + 1; 
    }
}

void indices_k(unordered_map<int,int> &p, int h_in, int nz, int pad, vector<vector<int>> &k)
{
    int ind = 0;
    int ksize = k.size();

    int x_tam = h_in + nz + 2*pad;
    
    for (int i=0; i<ksize; i++) 
    {
        for (int j=0; j<ksize; j++)
        {
            p[ind] = k[i][j];
            ind++;
        }
        ind += x_tam - ksize;
    }
}

vector<vector<int>> deconv(unordered_map<int, int> &px, unordered_map<int, int> &pk, int h_in, int ksize, int s)
{
    int h_out = (h_in-1)*s + ksize;
    vector<vector<int>> y(h_out, vector<int>(h_out));
    int sum;
    int ind = 0;
    for (int i=0; i <h_out;i++)
    {
        for(int j=0; j<h_out; j++)
        {
            sum = 0;
            for (auto k=pk.begin(); k!=pk.end(); k++) 
            {
                auto itx = px.find(k->first + ind);
                if (itx != px.end()) sum += k->second * itx->second;
            }
            if (j != h_out - 1) ind++;
            y[i][j] = sum;
        }
        ind += ksize;
    }
    return y;
}


vector<vector<int>> deconv_mejor(vector<vector<int>> &x, vector<vector<int>> &k, int s, int p)
{
    int h_in = x.size();
    int w_in = x[0].size();
    int h_k = k.size();
    int w_k = k[0].size();

    int h_out = h_in + s + h_k - 2;
    int w_out = w_in + s + w_k - 2;

    vector<vector<int>> y (h_out, vector<int>(w_out, 0));

    for (int i=0; i<h_in; i++)
    {
        for(int j=0; j<w_in; j++)
        {
            for(int ik=0; ik<h_k; ik++)
            {
                for(int jk=0; jk<w_k; jk++)
                {
                    y[ik + s*i][jk + s*j] += k[ik][jk]*x[i][j];
                }
            }
        }
    }
    return y;

}