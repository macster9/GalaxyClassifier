import matplotlib.pyplot as plt
import os

if __name__ == "__main__":

    os.environ['HOME'] = 'C:/Users/thoma/marvin'
    os.environ['SAS_BASE_DIR'] = os.path.join(os.getenv("HOME"), 'sas')
    print("Ignore following __warnings__:")
    plt.pause(0.1)
    from data_pipeline.queries import query_single
    query_single()
