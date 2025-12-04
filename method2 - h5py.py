import segyio
import numpy as np
import pandas as pd  
import h5py          
import json
import os
import warnings
import traceback
import time

warnings.filterwarnings('ignore')

# ----------------------------
# Config
# ----------------------------
# IMPORTANT: Replace with the actual path to your SEGY file
segy_file = r"C:\Users\USER\Desktop\seismic2\segy files\cropped_sobel_filter_similarity_Penobscot_kxky_pass2_broadband_26Dec19 - Copy.segy"
output_folder = r"C:\Users\USER\Desktop\seismic2\final compression store"

# Create output directories if they don't exist
os.makedirs(output_folder, exist_ok=True)

# ----------------------------
# Utility Functions
# ----------------------------

def fmt_bytes(size_bytes):
    """Formats bytes into KB, MB, GB, etc. for readability."""
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB")
    i = int(np.floor(np.log(size_bytes) / np.log(1024)))
    p = np.power(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_name[i]}"

def read_segy_data(path):
    """
    Reads all trace data and specific headers.
    """
    print(f"Opening SEGY file: {path}")
    try:
        with segyio.open(path, "r", ignore_geometry=True) as f:
            f.mmap()  # Use mmap for efficiency
            n_traces = f.tracecount
            n_samples = f.samples.size
            samples_array = np.array(f.samples) 
            
            if n_traces == 0 or n_samples == 0:
                print("ERROR: File contains 0 traces or 0 samples.")
                return None, None, None, None, None, None, None

            print(f"Total traces: {n_traces:,}")
            print(f"Samples per trace: {n_samples:,}")

            inlines, xlines, xs, ys, zs = [], [], [], [], []
            traces = np.empty((n_traces, n_samples), dtype=np.float32)
            
            print("Reading trace headers and data...")
            for i in range(n_traces):
                try:
                    header = f.header[i]
                    
                    inline = header.get(segyio.TraceField.INLINE_3D, 0)
                    xline = header.get(segyio.TraceField.CROSSLINE_3D, 0)
                    
                    # Fallback for 2D or missing headers
                    if inline == 0 and xline == 0:
                        grid_size = int(np.sqrt(n_traces))
                        if grid_size > 0 and (grid_size * grid_size) == n_traces:
                            inline = (i // grid_size) + 1
                            xline = (i % grid_size) + 1
                        else:
                            inline = i + 1
                            xline = 1
                    
                    try:
                        x_coord = header.get(segyio.TraceField.CDP_X, 0)
                        y_coord = header.get(segyio.TraceField.CDP_Y, 0)
                        if x_coord == 0 and y_coord == 0:
                            x_coord = header.get(segyio.TraceField.SourceX, 0)
                            y_coord = header.get(segyio.TraceField.SourceY, 0)
                    except Exception:
                        x_coord = 0
                        y_coord = 0
                        
                    z_coord = header.get(segyio.TraceField.DelayRecordingTime, 0)
                    trace_data = np.array(f.trace[i], dtype=np.float32)
                    trace_data = np.nan_to_num(trace_data, nan=0.0, posinf=0.0, neginf=0.0)

                    traces[i] = trace_data
                    inlines.append(inline)
                    xlines.append(xline)
                    xs.append(x_coord)
                    ys.append(y_coord)
                    zs.append(z_coord)

                except Exception as e:
                    print(f"Warning: Could not process trace {i}: {e}")
                    traces[i] = np.zeros(n_samples, dtype=np.float32)
                    inlines.append(np.nan)
                    xlines.append(np.nan)
                    xs.append(0)
                    ys.append(0)
                    zs.append(0)
            
        print("Data reading complete.")
        return traces, np.array(inlines), np.array(xlines), np.array(xs), np.array(ys), np.array(zs), samples_array
        
    except Exception as e:
        print(f"FATAL ERROR during SEGY read: {e}")
        return None, None, None, None, None, None, None

def extract_and_save_headers(segy_path, output_json_path):
    """
    Extracts all trace headers and saves them as a JSON file.
    """
    print(f"\nExtracting headers to {output_json_path}...")
    header_list = []
    with segyio.open(segy_path, "r", ignore_geometry=True) as f:
        f.mmap()
        for i in range(f.tracecount):
            hd = f.header[i]
            hdr_dict = {str(key): val for key, val in hd.items()}
            header_list.append(hdr_dict)
            
    with open(output_json_path, "w") as f:
        json.dump(header_list, f, indent=2)
    print("Header extraction complete.")

def print_trace_statistics(title, traces_data, inlines, xlines, xs, ys, samples):
    """
    Calculates and prints descriptive statistics for the data.
    """
    print(f"\n--- {title} ---")
    
    try:
        n_traces, n_samples = traces_data.shape
        total_samples = n_traces * n_samples

        print(f"Calculating statistics for {total_samples:,} total samples (memory-efficiently)...")

        stats_list = []

        # 1. --- 'index' column ---
        n = float(total_samples)
        index_stats = {
            'index': 'index', 'count': n, 'mean': (n - 1) / 2.0,
            'std': np.sqrt( ( (n**2) - 1) / 12.0 * (n / (n - 1)) ),
            'min': 0.0, '25%': (n - 1) * 0.25, '50%': (n - 1) * 0.5,
            '75%': (n - 1) * 0.75, 'max': n - 1
        }
        stats_list.append(index_stats)

        # 2. --- 'Inline', 'Xline', 'X', 'Y' columns ---
        for name, data_array in [('Inline', inlines), ('Xline', xlines), ('X', xs), ('Y', ys)]:
            s = pd.Series(data_array).describe()
            stat_dict = s.to_dict()
            stat_dict['index'] = name
            stat_dict['count'] = float(total_samples)
            stats_list.append(stat_dict)

        # 3. --- 'Z' column ---
        s = pd.Series(samples).describe()
        stat_dict = s.to_dict()
        stat_dict['index'] = 'Z'
        stat_dict['count'] = float(total_samples)
        stats_list.append(stat_dict)
        
        # 4. --- 'Amp' column ---
        # Use numpy for speed on large arrays
        p25, p50, p75 = np.percentile(traces_data, [25, 50, 75])
        amp_stats = {
            'index': 'Amp', 'count': float(total_samples),
            'mean': np.mean(traces_data), 'std': np.std(traces_data, ddof=1),
            'min': np.min(traces_data), '25%': p25, '50%': p50,
            '75%': p75, 'max': np.max(traces_data)
        }
        stats_list.append(amp_stats)

        # --- Assemble the final DataFrame ---
        result_df = pd.DataFrame(stats_list).set_index('index')
        column_order = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
        row_order = ['index', 'Inline', 'Xline', 'X', 'Y', 'Z', 'Amp']
        result_df = result_df[column_order].reindex(row_order)

        # --- Set pandas options to format numbers ---
        pd.set_option('display.float_format', lambda x: f'{x:,.2f}')
        pd.set_option('display.width', 1000)
        pd.set_option('display.max_columns', None)
        
        print(result_df.to_string())

    except Exception as e:
        print(f"Error during statistics calculation: {e}")
        traceback.print_exc()

# ----------------------------
# Main Execution
# ----------------------------
def main():
    try:
        # --- 1. Extract Trace Data (and some headers to memory) ---
        print("\n" + "="*50)
        print("--- READ TRACE DATA ---")
        print("="*50)
        print(f"Reading data from {segy_file}...")
        traces, inlines, xlines, xs, ys, zs, samples = read_segy_data(segy_file)
        
        if traces is None:
            raise Exception("Failed to read SEGY data. Halting execution.")
            
        original_trace_bytes = traces.nbytes
        original_file_size = os.path.getsize(segy_file)
        
        print(f"\nSuccessfully extracted {traces.shape[0]} traces with {traces.shape[1]} samples each.")
        print(f"Original traces shape (2D): {traces.shape}")

        # --- 2. Extract and Save Header Metadata ---
        print("\n" + "="*50)
        print("--- SAVE HEADER METADATA ---")
        print("="*50)
        header_json_path = os.path.join(output_folder, "trace_headers.json")
        extract_and_save_headers(segy_file, header_json_path)
        print(f"All header metadata saved to: {header_json_path}")

        # --- 3. Print All Statistics ---
        print("\n" + "="*50)
        print("--- ORIGINAL FILE STATISTICS ---")
        print("="*50)
        print(f"Original file size: {fmt_bytes(original_file_size)}")
        print(f"Original trace data size: {fmt_bytes(original_trace_bytes)}")
        
        print_trace_statistics("Original Data Statistics", traces, inlines, xlines, xs, ys, samples)

        # --- 4. Compress Trace Data using HDF5 ---
        print("\n" + "="*50)
        print("--- COMPRESS TRACE DATA (HDF5) ---")
        print("="*50)
        hdf5_file_path = os.path.join(output_folder, "compressed_traces.h5")
        print(f"Saving compressed traces to {hdf5_file_path} using HDF5 (GZIP)...")
        
        start_write = time.time()
        with h5py.File(hdf5_file_path, 'w') as hf:
            # Create a dataset within the HDF5 file
            # compression="gzip" is standard. "lzf" is faster but less efficient.
            # chunks=(1000, n_samples) allows efficient partial reading later
            hf.create_dataset(
                'amplitude_data', 
                data=traces, 
                compression="gzip", 
                compression_opts=4, # Level 4 is a good balance of speed/size
                chunks=(1000, traces.shape[1])
            )
            # Attach basic metadata to the file for reference
            hf.attrs['original_file'] = os.path.basename(segy_file)
            hf.attrs['creation_date'] = time.ctime()

        compressed_file_size = os.path.getsize(hdf5_file_path)
        print(f"Write time: {time.time() - start_write:.2f} seconds")
        print(f"Compressed file size: {fmt_bytes(compressed_file_size)}")
        
        if compressed_file_size > 0:
            compression_ratio = original_trace_bytes / compressed_file_size
            print(f"Compression ratio: {compression_ratio:.2f}x")
        
        print(f"Original trace data size: {fmt_bytes(original_trace_bytes)}")

        # --- 5. Load Compressed Data and Print Statistics ---
        print("\n" + "="*50)
        print("--- LOAD HDF5 DATA AND VERIFY STATISTICS ---")
        print("="*50)
        
        # Open the HDF5 file to verify we can read it back
        with h5py.File(hdf5_file_path, 'r') as hf:
            # Load the dataset back into memory as a numpy array
            # Note: For massive files, you wouldn't load [:] all at once, 
            # but we do it here to pass it to your existing statistics function.
            compressed_traces = hf['amplitude_data'][:]
            
            print(f"Loaded compressed traces shape: {compressed_traces.shape}")
            print_trace_statistics("HDF5 Decompressed Statistics", compressed_traces, inlines, xlines, xs, ys, samples)

    except FileNotFoundError:
        print(f"\nERROR: SEGY file not found at path: {segy_file}")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
