import subprocess
import random


def get_free_port():
    """
    Returns a free port.
    """
    # get random in between 2000 and 3000 divisble by 5
    port = random.randint(2000, 3000)
    port = port - (port % 5)

    # port = 2000
    port_free = False

    while not port_free:
        try:
            pid = int(
                subprocess.check_output(
                    f"lsof -t -i :{port} -s TCP:LISTEN",
                    shell=True,
                ).decode("utf-8")
            )
            # print(f'Port {port} is in use by PID {pid}')
            port += 5

        except subprocess.CalledProcessError:
            port_free = True
            print(port)
            # print(f'Port {port} is free')


if __name__ == "__main__":
    get_free_port()
