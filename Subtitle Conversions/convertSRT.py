import re

def convert_to_srt(input_filename, output_filename):
    def format_timestamp(seconds):
        milliseconds = int((seconds % 1) * 1000)
        seconds = int(seconds)
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        return f'{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}'

    # Open the input and output files
    with open(input_filename, 'r') as infile, open(output_filename, 'w') as outfile:
        subtitle_index = 1  # Initialize subtitle index
        
        # Read each line in the input file
        for line in infile:
            # Split the line into individual subtitle entries
            entries = line.strip().split('}\\n')
            for entry in entries:
                if entry:
                    entry = entry.strip() + '}'  # Add the closing brace back
                    # Match the entry with the expected format
                    match = re.match(r"\{'timestamp': \((\d+\.\d+), (\d+\.\d+)\), 'text': (.+?)\}\s*", entry)
                    if match:
                        start, end, text = match.groups()
                        # Convert timestamps to SRT format
                        start_srt = format_timestamp(float(start))
                        end_srt = format_timestamp(float(end))
                        # Write the subtitle entry to the output file
                        outfile.write(f'{subtitle_index}\n{start_srt} --> {end_srt}\n{text.strip(" '\"")}\n\n')
                        subtitle_index += 1  # Increment the subtitle index

if __name__ == "__main__":
    # Replace with your input and output file paths
    input_filename = './output/aud2txt.txt'
    output_filename = './output/output.srt'
    convert_to_srt(input_filename, output_filename)
