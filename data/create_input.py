import os
import re
import time
import logging
import argparse


def setup_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('[%(levelname)s] %(asctime)s - %(message)s')
    stream_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)


def get_args():
    parser = argparse.ArgumentParser(prog=os.path.splitext(os.path.basename(__file__))[0])
    parser.add_argument("--dir", default='bach', help="Generate input for directory.")

    return vars(parser.parse_args())


def main():
    setup_logger()
    args = get_args()

    logging.info("Input generation for {dir} directory started.".format(dir=args['dir']))
    start_time = time.time()

    with open(args['dir'] + "/input.txt", 'wb') as fw:
        file_list = os.listdir(args['dir'])
        file_list.sort()

        for filename in file_list:
            if not filename.endswith(".abc"):
                continue

            with open(os.path.join(args['dir'], filename)) as f:
                if re.search(r"M:\s*(?:[^4]/|/[^4])", f.read(), flags=re.MULTILINE | re.IGNORECASE):
                    continue

            with open(os.path.join(args['dir'], filename)) as f:
                for line in f.readlines():
                    if line.startswith("%") or re.match(r".\s*:", line):
                        continue

                    fw.write(line)

    end_time = time.time()
    logging.info("Input generation completed in {time_diff} seconds.".format(time_diff=end_time - start_time))

if __name__ == '__main__':
    main()
