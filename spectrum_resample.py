import matplotlib.pyplot as plt
import matplotlib
from scipy import interpolate

kind_list = ["linear", "nearest","nearest-up","zero","slinear",
            "quadratic","cubic","previous","next"]

def spectrum_resample(ev,x,ev2,kind):
    #https://watlab-blog.com/2019/09/19/resampling/
    # 補間関数fを作成
    f = interpolate.interp1d(ev, x, kind=kind)

    # 補間した結果からリサンプリング波形を生成
    num = len(ev2)
    t0 = 0
    tf = ev2[-1]
    num = len(ev2)
    ev_resample = ev2
    x_resample = f(ev_resample)              # f(t)

    #plt.plot(ev,x)
    fig = plt.figure(figsize=(15,5))
    plt.plot(ev_resample,x_resample, label = kind)
    plt.legend()
    #plt.show()
    return(x_resample)

x_resample = spectrum_resample(ev,x,ev2,kind="cubic")

data = np.vstack([ev2,x_resample])
df = pd.DataFrame(data = data.T,
                  columns=["ev","input"])
df.to_csv("spectrum1.csv")
