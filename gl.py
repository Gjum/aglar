from collections import defaultdict
from ctypes import cast, c_char_p
import io
import random
import select
from threading import Thread
import time
from urllib import request

import pyglet
from pyglet import gl
from pyglet.window import key

import agarnet
from agarnet.utils import find_server, moz_headers, special_names
from agarnet.vec import Vec

from shader import Shader

# TODO stats:
#   food+cells eaten
#   time alive
#   highest mass
#   leaderboard time
#   top position

skin_texture_cache = {}

def get_skin_tex(name, synchronous=False):
    name = name.lower()
    if name not in special_names:
        return None

    if name not in skin_texture_cache:
        # load in separate thread, return None for now
        skin_texture_cache[name] = None

        def loader():
            opener = request.build_opener()
            opener.addheaders = moz_headers
            skin_url = 'http://agar.io/skins/%s.png' % request.quote(name)
            skin_data = io.BytesIO(opener.open(skin_url).read())
            skin_texture = pyglet.image.load('.png', skin_data).get_texture()
            print('Loaded skin', name)
            skin_texture_cache[name] = skin_texture

        if synchronous:
            loader()
        else:
            t = Thread(target=loader)
            t.setDaemon(True)
            t.start()

    return skin_texture_cache[name]


foo_size = Vec(1920, 1080)  # size of Zeach's monitor
win_size = foo_size / 2

target_win = Vec()  # window pos of mouse, from top left

window = pyglet.window.Window(*map(int, win_size), caption="aGLar",
                              resizable=True, visible=False)

shader_circle_tex = Shader(vert=[open('shaders/circle_tex.vert').read()],
                           frag=[open('shaders/circle_tex.frag').read()])

# set the correct texture unit xxx why?
# shader_circle_tex.bind()
# shader_circle_tex.uniformi('tex0', 0)
# shader_circle_tex.unbind()

gl.glClearColor(.2,.2,.2,1)
gl.glEnable(gl.GL_BLEND)
gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)


@window.event
def on_resize(width, height):
    gl.glViewport(0, 0, width, height)
    # projection gets recalculated before next draw
    win_size.set(width, height)
    leaderboard_label.x, leaderboard_label.y = win_size
    return pyglet.event.EVENT_HANDLED


def rect_corners(top_left, bottom_right):
    l, t = top_left
    r, b = bottom_right
    t = client.player.world.center.y - t
    b = client.player.world.center.y - b
    return (
        l, t,
        r, t,
        r, b,
        l, b,
    )

def rect(x, y, r, ry=None):
    y = client.player.world.center.y - y
    if ry is None:
        ry = r
    return (
        x-r, y-ry,
        x+r, y-ry,
        x+r, y+ry,
        x-r, y+ry,
    )

class CustomCell(agarnet.world.Cell):
    def __init__(self, **kwargs):
        super(CustomCell, self).__init__(**kwargs)

        self.name_label = pyglet.text.Label(
            self.name, anchor_x='center', anchor_y='bottom',
            x=self.pos.x, y=client.world.center.y - self.pos.y,
            font_size=self.font_size)

        self.info_label = pyglet.text.Label(
            self.info_text, anchor_x='center', anchor_y='top',
            x=self.pos.x, y=client.world.center.y - self.pos.y,
            font_size=self.font_size)

    def draw_text(self, name_batch=None, info_batch=None):
        if self.is_food or self.is_ejected_mass:
            return

        if self.name and name_batch:
            self.name_label.begin_update()
            self.name_label.batch = name_batch
            self.name_label.font_size = self.font_size
            self.name_label.x = self.pos.x
            self.name_label.y = client.world.center.y - self.pos.y
            self.name_label.text = self.name
            self.name_label.end_update()

        if info_batch:
            self.info_label.begin_update()
            self.info_label.batch = info_batch
            self.info_label.font_size = self.font_size
            self.info_label.x = self.pos.x
            self.info_label.y = client.world.center.y - self.pos.y
            self.info_label.text = self.info_text
            self.info_label.end_update()

    @property
    def font_size(self):
        return max(15/client.player.scale, self.size/5)

    @property
    def info_text(self):
        info = '%i' % self.mass
        masses = [c.mass for c in client.player.own_cells if c.mass > 0]
        if self not in client.player.own_cells and masses:
            pct_min = 100 * self.mass / min(masses)
            pct_max = 100 * self.mass / max(masses)
            info += ' %i%%' % pct_min
            if pct_min != pct_max:
                info += ' %i%%' % pct_max
        return info

class CC(CustomCell):  # xxx testing
    def __init__(self, pos, size, color, name):
        super().__init__()
        self.pos = Vec(pos)
        self.size = size
        self.color = color
        self.name = name


def screen_to_world(screen_pos):
    scale = max(*win_size.vdiv(foo_size)) * client.player.scale
    return Vec(screen_pos - win_size / 2) / scale + client.player.center

def set_world_projection():
    center = client.player.center
    horz_vp = min(foo_size.x, foo_size.y * win_size.x / win_size.y)
    vert_vp = min(foo_size.y, foo_size.x * win_size.y / win_size.x)
    s = client.player.scale * 2.  # window border is /2 from center

    gl.glMatrixMode(gl.GL_PROJECTION)
    gl.glLoadIdentity()
    gl.glOrtho(-horz_vp / s, horz_vp / s,
               -vert_vp / s, vert_vp / s,
               -1,1)
    gl.glTranslatef(-center.x, center.y, 0)
    gl.glMatrixMode(gl.GL_MODELVIEW)

def set_hud_projection():
    gl.glMatrixMode(gl.GL_PROJECTION)
    gl.glLoadIdentity()
    gl.glOrtho(0, win_size.x, 0, win_size.y, -1, 1)
    gl.glMatrixMode(gl.GL_MODELVIEW)

class CellDrawer(object):
    """
    Collects all cells and groups them into batches to increase performance.
    """

    def __init__(self):
        self.data = defaultdict(lambda: ([], []))
        self.cells = []

    def add_cell(self, cell):
        self.cells.append(cell)

        texture = get_skin_tex(cell.name, synchronous=True)  # xxx async pliz
        vertices, colors = self.data[texture]

        vertices.extend(rect(*cell.pos, r=cell.size))

        color = list(cell.color)
        color.append(float(not texture))  # mark textured cells
        colors.extend(color*4)

    def draw_all_cells(self):
        set_world_projection()

        # circles
        shader_circle_tex.bind()
        for texture in self.data:
            self.draw_similar_cells(texture)
        shader_circle_tex.unbind()

        # names + info
        cell_name_batch = pyglet.graphics.Batch()
        cell_info_batch = pyglet.graphics.Batch()
        for cell in self.cells:
            cell.draw_text(show_cell_names and cell_name_batch,
                           show_cell_info and cell_info_batch)
        cell_name_batch.draw()
        cell_info_batch.draw()

        # data changes every frame
        self.data.clear()
        self.cells.clear()

    def draw_similar_cells(self, texture):
        vertices, colors = self.data[texture]

        # convert all cell data into one batch
        batch = pyglet.graphics.Batch()
        num_cells = len(vertices) // 8
        batch.add(num_cells*4, gl.GL_QUADS, None,
                  ('v2f', vertices),
                  ('c4f', colors),
                  ('t2f', (0,0, 1,0, 1,1, 0,1)*num_cells))

        if texture:
            gl.glBindTexture(texture.target, texture.id)

        # shader.uniformf('pixel', 1.0/width, 1.0/height) xxx

        # draw all cells at once
        batch.draw()

        if texture:
            gl.glBindTexture(texture.target, 0)  # unbind


def set_minimap_projection():
    # third of window height, bottom-right corner
    y = 3 * client.world.size.y
    x = y / win_size.y * win_size.x
    top_left = client.world.top_left

    gl.glMatrixMode(gl.GL_PROJECTION)
    gl.glLoadIdentity()
    gl.glOrtho(0, x, -y, 0, -1, 1)
    gl.glTranslatef(-top_left.x, top_left.y, 0)
    gl.glMatrixMode(gl.GL_MODELVIEW)

def draw_minimap():
    if not client.world.size: return

    set_minimap_projection()

    vertices, colors = [],[]
    num_cells = 0
    for cell in client.player.world.cells.values():
        if cell.is_food or cell.is_ejected_mass: continue
        num_cells += 1

        vertices.extend(rect(*cell.pos, r=cell.size))

        color = list(cell.color)
        color.append(1.0)  # xxx mark as hollow
        colors.extend(color*4)

    batch = pyglet.graphics.Batch()
    batch.add(num_cells*4, gl.GL_QUADS, None,
              ('v2f', vertices),
              ('c4f', colors),
              ('t2f', (0,0, 1,0, 1,1, 0,1)*num_cells))

    world = client.player.world
    # background
    pyglet.graphics.draw(4, gl.GL_QUADS,
         ('v2f', rect_corners(world.top_left, world.bottom_right)),
         ('c4f', tuple([.2,.2,.2, .8]*4)))
    # border
    gl.glColor4f(.3,.3,.3, 1)
    pyglet.graphics.draw(4, gl.GL_LINE_LOOP,
         ('v2f', rect_corners(world.top_left, world.bottom_right)))

    # outline the area visible in window
    pyglet.graphics.draw(4, gl.GL_LINE_LOOP, ('v2f', rect_corners(
        screen_to_world(Vec(0,0)), screen_to_world(win_size))))

    shader_circle_tex.bind()
    batch.draw()
    shader_circle_tex.unbind()


@window.event
def on_draw():
    window.clear()

    set_world_projection()

    # world border
    gl.glLineWidth(1)
    pyglet.graphics.draw(4, gl.GL_LINE_LOOP,
        ('v2f', rect_corners(client.player.world.top_left,
                             client.player.world.bottom_right)),
        ('c4f', tuple([1.,1.,1., .1]*4)))

    # cells

    # gather cell data, draw later using batch
    cell_drawer = CellDrawer()

    # xxx testing: frame with lying eight of varying nr of cells in window center

    import math
    eight_radius = 1920/2
    n = 0  # used at the end for perf

    def spam(amount, size):
        nonlocal n
        n += amount
        for i in range(amount):
            angle = i * 6.283 / amount  # normalize to circle
            angle *= .99  # gap
            angle += time.time()  # rotate over time
            pos = Vec(eight_radius * math.cos(angle),
                      1080/1920 * eight_radius * math.sin(2*angle))  # lying eight
            pos += client.player.center
            cell_drawer.add_cell(CC(pos, size, (1.,.5,0.),
                                    ('brazil', 'doge', 'sdfsd')[i % 3]))

    # spam(int(time.time()*50) % 500, 200)
    # spam(9, 150)
    # spam(7, 1500)
    # spam(200, 200)
    # spam(500, 5)

    # the real cells
    for c in sorted(client.player.world.cells.values()):
        cell_drawer.add_cell(c)

    cell_drawer.draw_all_cells()

    draw_minimap()

    # HUD

    set_hud_projection()

    # mass graph
    if mass_graph:
        mass_verts = []
        dx = win_size.x / 3 / len(mass_graph)
        ay = win_size.y / 3 / max(mass_graph)
        for i, mass in enumerate(mass_graph):
            mass_verts.append(win_size.x * 2/3 + i * dx)
            mass_verts.append(mass * ay)
        gl.glLineWidth(3)
        gl.glColor4f(0,0,1, 1)
        pyglet.graphics.draw(len(mass_graph), gl.GL_LINE_STRIP, ('v2f', mass_verts))
        gl.glLineWidth(1)

    # leaderboard
    leaderboard_text = '\n'.join('%s (%i)' % (name,pos+1) for pos,(id,name) in
                                 enumerate(client.world.leaderboard_names))
    if leaderboard_label.text != leaderboard_text:
        leaderboard_label.text = leaderboard_text  # causes recalculation
    leaderboard_label.draw()

    # measure performance
    if show_fps:
        global last_frame
        now = time.time()
        frame_fps = 1/(now - last_frame)
        last_frame = now

        pyglet.text.Label(
            '%.2f' % frame_fps, font_size=20, color=(255,255,0,255),
            x=0, y=win_size.y,
            anchor_x='left', anchor_y='top').draw()

show_cell_names = True
show_cell_info = True
show_fps = True
last_frame = time.time()

leaderboard_label = pyglet.text.Label(
    '', x=win_size.x, y=win_size.y, width=win_size.x, font_size=15,
    multiline=True, align='right', anchor_x='right', anchor_y='top')

@window.event
def on_mouse_motion(x, y, dx, dy):
    target_win.set(x, win_size.y-y)

@window.event
def on_key_press(symbol, modifiers):
    try: print('key', chr(symbol), hex(symbol), symbol)
    except: pass

    if symbol == key.S:
        client.send_spectate()
    elif symbol == key.Q:
        client.send_spectate_toggle()
    elif symbol == key.R:
        client.send_respawn()
        mass_graph.clear()
    elif symbol == key.W:
        client.send_target(*screen_to_world(target_win))
        client.send_shoot()
    elif symbol == key.SPACE:
        client.send_split()
    elif symbol == key.C:
        client.disconnect()
        client.connect(*find_server())
    elif symbol == key.N:
        global show_cell_names
        show_cell_names = not show_cell_names
    elif symbol == key.I:
        global show_cell_info
        show_cell_info = not show_cell_info
    elif symbol == key.F3:
        global show_fps
        show_fps = not show_fps


class Subscriber(object):
    """Base class for event handlers via on_*() methods."""

    def __getattr__(self, func_name):
        # still throw error when not getting an `on_*()` attribute
        if 'on_' != func_name[:3]:
            raise AttributeError("'%s' object has no attribute '%s'"
                                 % (self.__class__.__name__, func_name))
        return lambda *args, **kwargs: None  # default handler does nothing

    def on_message_error(self, err):  # xxx
        raise ValueError(err)

    def on_world_update_post(self):
        if client.player.is_alive:
            mass_graph.append(client.player.total_mass)

mass_graph = []

def process_packets(client, timeout=0.001):
    # process all waiting packets, but do not block
    while True:
        r, w, e = select.select((client.ws.sock,), (), (), timeout)
        if r:
            client.on_message()
        elif e:
            client.subscriber.on_sock_error(e)
        else:
            break  # no packet received

def on_tick(dt):
    client.send_target(*screen_to_world(target_win))
    process_packets(client)


sub = Subscriber()
client = agarnet.client.Client(sub)
client.world.cell_class = CustomCell

client.player.nick = random.choice(special_names)
print('player name:', client.player.nick)

try:
    client.connect(*find_server())
except ConnectionResetError:
    client.connect(*find_server())

pyglet.clock.schedule_interval(on_tick, 1./60.)
window.set_visible(True)
pyglet.app.run()
