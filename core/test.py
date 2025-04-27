from matplotlib import pyplot as plt
from scipy.stats import pearsonr

from core.builder import generate_code
from core.executor import AbstractPlatform
from core.strategy import RandomSelectionStrategy
from core.value import I32


def main():
    fuels = []
    response_times = []
    response_times2 = []
    code_size = []
    plt.ion()
    fig, (ax, ax2, ax3) = plt.subplots(3)
    hist, edges, _ = ax.hist(fuels, bins=100)
    #Labels
    ax.set_xlabel("Fuel")
    ax.set_ylabel("Count")
    scatter = ax2.scatter(fuels, response_times)
    ax2.set_xlabel("Fuel")
    ax2.set_ylabel("Response time")
    scatter2 = ax2.scatter(fuels, response_times2)
    ax2.set_xlabel("Fuel")
    ax2.set_ylabel("Response time")
    ax3.scatter(fuels, code_size)
    ax3.set_xlabel("Fuel")
    ax3.set_ylabel("Code size")
    cpu_only_platform = AbstractPlatform("Test Platform", cpu_cycles_per_second=4 * 10e9)
    gpu_only_platform = AbstractPlatform("Test Platform", cpu_cycles_per_second=2 * 10e9,
                                         gpu_cycles_per_second=1 * 10e9)
    for result in generate_code(19515-1, min_byte_code_size=20, max_byte_code_size=50, min_fuel=0,
                                max_fuel=2000, verbose=True, selection_strategy=RandomSelectionStrategy(),output_types=[I32]):
        print(result.code_str)
        print("Fuel:", result.abstract_run_result.fuel)
        print("Ext:", result.abstract_run_result.ext_resources)
        print("Byte code length:", len(result.byte_code))
        print("Return value:", result.abstract_run_result.return_values)
        print("Memory:", len(result.initial_memory))
        print("Response time CPU only:", cpu_only_platform.calculate_response_time(result.abstract_run_result))
        print("Response time CPU and GPU:", gpu_only_platform.calculate_response_time(result.abstract_run_result))
        response_times.append(cpu_only_platform.calculate_response_time(result.abstract_run_result))
        response_times2.append(gpu_only_platform.calculate_response_time(result.abstract_run_result))
        fuels.append(result.abstract_run_result.fuel)
        code_size.append(len(result.byte_code))
        ax.cla()
        ax.hist(fuels, bins=100)
        ax.set_xlabel("Fuel")
        ax.set_ylabel("Count")
        ax2.cla()
        ax2.scatter(fuels, response_times, label="CPU")
        ax2.scatter(fuels, response_times2, label="GPU")
        ax2.set_xlabel("Fuel")
        ax2.set_ylabel("Response time")
        ax2.legend()
        #Set x log
        ax2.set_yscale('log')
        ax3.cla()
        ax3.scatter(fuels, code_size)
        ax3.set_xlabel("Fuel")
        ax3.set_ylabel("Code size")
        #Set max x and log
        fig.canvas.draw()
        fig.canvas.flush_events()

        if len(fuels) > 1:
            corr, _ = pearsonr(fuels, response_times)
            print('Pearsons correlation fuels response time CPU: %.3f' % corr)
            corr, _ = pearsonr(fuels, response_times2)
            print('Pearsons correlation fuels response time CPU and GPU: %.3f' % corr)
            corr, _ = pearsonr(code_size, fuels)
            print('Pearsons correlation fuels code size: %.3f' % corr)
        plt.pause(0.1)



if __name__ == "__main__":
    main()
