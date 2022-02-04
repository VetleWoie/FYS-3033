import time

def progress_bar(iteration, total, done="=", not_done=" ",show_total=False, show_percentage=True, length=30, new_line=True):
    percentage = iteration/total
    num_done = int(percentage*length)
    done = done*(num_done)
    not_done = not_done*(length-num_done)
    out = f"[{done}{not_done}]"
    if show_total:
        out += f"|{iteration}/{total}"
    if show_percentage:
        out += "|"+str(percentage*100)+"%"
    print(out,end="\n" if new_line else"\r")