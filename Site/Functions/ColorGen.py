import sys
EPSILON = sys.float_info.epsilon


class ColorGen(object):
    def __init__(self, minval, maxval, colors_hex):
        self.minval = minval
        self.maxval = maxval
        self.colors_hex = colors_hex
        self.colors_rgb = [hex_to_RGB(color) for color in colors_hex]
        self.n_colors = len(colors_hex)
        self.range = float(self.maxval - self.minval)

    def convert_to_rgb(self, val):
        i_f = float(val - self.minval) / self.range * (self.n_colors - 1)
        i, f = int(i_f // 1), i_f % 1
        if f < EPSILON:
            return self.colors_rgb[i]
        else:
            (r1, g1, b1), (r2, g2, b2) = self.colors_rgb[i], self.colors_rgb[i + 1]
            return (int(r1 + f * (r2 - r1)), int(g1 + f * (g2 - g1)), int(b1 + f * (b2 - b1)))

    def convert_to_hex(self, val):
        return RGB_to_hex(self.convert_to_rgb(val))


def RGB_to_hex(RGB):
    color = '#'
    for num in RGB:
        color += str(hex(num))[-2:].replace('x', '0').upper()
    return color


# 16进制颜色格式颜色转换为RGB格式
def hex_to_RGB(hex):
    r = int(hex[1:3], 16)
    g = int(hex[3:5], 16)
    b = int(hex[5:7], 16)
    return r, g, b


if __name__ == '__main__':
    color_gen = ColorGen(minval=-1, maxval=1, colors_hex=['#313695', '#4575b4', '#74add1', '#abd9e9', '#e0f3f8', '#ffffbf', '#fee090', '#fdae61', '#f46d43', '#d73027', '#a50026'])
