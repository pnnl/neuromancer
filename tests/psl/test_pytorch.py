from neuromancer.psl.autonomous import systems as systems1
# from autonomous import systems  as systems2
import timeit
import numpy as np

if __name__ == "__main__":
    """
    Fewer lines of code.
    Added functionality with torch backend.
    Backwards compatible.
    3X faster than original for original functionality.
    Order of magnitude slower for new functionality with torch backend.
    """
    raise RuntimeError("""I had to disable this test case, it seems that the script is referencing a local autonomous file that does not exist on the master branch""")
    n = 20
    nsim = 100
    diffs_pt = []
    diffs_np = []
    diffs_pp = []
    for sys1, sys2 in zip(systems1.values(), systems2.values()):
        print(sys2)
        s1 = sys1(nsim=nsim)
        s2 = sys2(backend='numpy', nsim=nsim)
        s3 = sys2(backend='torch', nsim=nsim)
        s4 = sys2(backend='torch', nsim=nsim, requires_grad=True)

        data1 = s1.simulate(nsim=10)
        data2 = s2.simulate(nsim=10)
        data3 = s3.simulate(nsim=10)
        data4 = s4.simulate(nsim=10)

        print(f"{((data1['X']-data3['X'].numpy())**2).mean().item(): 2f}")
        assert abs(data1['X']-data2['X']).all() < 1e-6
        assert abs(data2['X']-data3['X'].numpy()).all() < 1e-6
        assert abs(data3['X']-data4['X']).all() < 1e-6

        time1 = timeit.timeit(s1.simulate, number=n)
        time2 = timeit.timeit(s2.simulate, number=n)
        time3 = timeit.timeit(s3.simulate, number=n)
        time4 = timeit.timeit(s4.simulate, number=n)
        diffs_np.append(time1/time2)
        diffs_pt.append(time2/time3)
        diffs_pp.append(time3/time4)

    print("New vs old np", np.array(diffs_np).mean())
    print("New np vs pt", np.array(diffs_pt).mean())
    print("New pt vs pt_grad", np.array(diffs_pp).mean())