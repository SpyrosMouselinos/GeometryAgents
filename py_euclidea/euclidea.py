import gi
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk, Gdk
from .constructions import *
from .tools import MoveTool, key_to_tool, tool_name_to_key



class Drawing(Gtk.Window):

    def __init__(self, env):
        super(Drawing, self).__init__()
        self.shift = np.array([0, 0])
        self.scale = 1
        self.mb_grasp = None

        self.set_env(env)
        self.darea = Gtk.DrawingArea()
        self.darea.connect("draw", self.on_draw)
        self.darea.set_events(Gdk.EventMask.BUTTON_PRESS_MASK |
                              Gdk.EventMask.KEY_PRESS_MASK |
                              Gdk.EventMask.SCROLL_MASK |
                              Gdk.EventMask.BUTTON1_MOTION_MASK |
                              Gdk.EventMask.BUTTON2_MOTION_MASK)
        self.add(self.darea)

        self.darea.connect("button-press-event", self.on_button_press)
        self.darea.connect("scroll-event", self.on_scroll)
        self.darea.connect("motion-notify-event", self.on_motion)
        self.connect("key-press-event", self.on_key_press)

        self.set_title("Drawing")
        self.resize(*(637, 490))
        self.set_position(Gtk.WindowPosition.CENTER)
        self.connect("delete-event", Gtk.main_quit)
        self.show_all()

    def set_env(self, env):
        self.env = env
        print("Enabled tools:")
        for name in sorted(self.env.enabled_tools):
            if name not in tool_name_to_key:
                print(name)
            key = tool_name_to_key[name]
            print("  {} ({})".format(name, key))
        assert (env.start_tool in env.enabled_tools)
        self.set_tool(tool_name_to_key[env.start_tool])

    def get_coor(self, e):
        return np.array([e.x, e.y]) / self.scale - self.shift

    def on_scroll(self, w, e):
        coor = self.get_coor(e)
        if e.direction == Gdk.ScrollDirection.DOWN:
            self.scale *= 0.9
        elif e.direction == Gdk.ScrollDirection.UP:
            self.scale /= 0.9
        print("zoom {}".format(self.scale))
        self.shift = np.array([e.x, e.y]) / self.scale - coor
        self.darea.queue_draw()

    def on_motion(self, w, e):
        if e.state & Gdk.ModifierType.BUTTON1_MASK:
            if isinstance(self.tool, MoveTool) and self.tool.grabbed is not None:
                step = self.tool.grabbed
                step.coor = self.get_coor(e)
                self.env.run_steps()
                self.tool.refresh(self.env)
                self.darea.queue_draw()
        if e.state & Gdk.ModifierType.BUTTON2_MASK:
            if self.mb_grasp is None: return
            self.shift = np.array([e.x, e.y]) / self.scale - self.mb_grasp
            self.darea.queue_draw()

    def on_draw(self, wid, cr):

        corners = np.array([
            [0, 0],
            [self.darea.get_allocated_width(), self.darea.get_allocated_height()],
        ]) / self.scale - self.shift
        cr.scale(self.scale, self.scale)
        cr.translate(*self.shift)
        cr.rectangle(*corners_to_rectangle(corners))
        cr.set_source_rgb(1, 1, 1)
        cr.fill()

        highlighted = self.tool.get_highlighted(self.env)
        goals = self.env.cur_goal()
        constructed = tuple(self.env.objs_of_type(GeoObject))
        constr_basic = []
        constr_desired = []
        for obj in constructed:
            if any(goal.identical_to(obj) for goal in goals):
                constr_desired.append(obj)
            else:
                constr_basic.append(obj)
        goals_remaining = [
            goal for goal in goals
            if not any(goal.identical_to(obj) for obj in constr_desired)
        ]

        for t in (PointSet, Point):
            for objs, color in (
                    (goals_remaining, (0.87, 0.86, 0.2)),
                    (constr_basic, (0, 0, 0)),
                    (constr_desired, (0.06, 0.58, 0.07)),
                    (highlighted, (0.7, 0, 1)),
            ):

                cr.set_source_rgb(*color)
                for obj in objs:
                    if isinstance(obj, t): obj.draw(cr, corners, self.scale)

    def on_key_press(self, w, e):
        keyval = e.keyval
        keyval_name = Gdk.keyval_name(keyval)
        if keyval_name in key_to_tool:
            self.set_tool(keyval_name)
            self.darea.queue_draw()
        elif keyval_name == 'r':
            print("Random")
            self.env.rnd_init()
            self.darea.queue_draw()
        elif keyval_name == 'BackSpace':
            print("BACK")
            self.env.pop()
            self.tool.refresh(self.env)
            print("Status:", self.env.check_goal())
            self.darea.queue_draw()
        elif keyval_name == "Tab":
            self.env.next_goal()
            self.darea.queue_draw()
        elif keyval_name == "Escape":
            Gtk.main_quit()
        else:
            return False

    def set_tool(self, key):
        tool, name = key_to_tool[key]
        if name not in self.env.enabled_tools:
            print("tool {} is not enabled on the current problem".format(name))
            return
        self.tool = tool
        print("Tool: {}".format(name))
        self.tool.initialize(self.env)

    def on_button_press(self, w, e):
        coor = self.get_coor(e)
        if e.button == 1 and self.tool is not None:
            if e.type != Gdk.EventType.BUTTON_PRESS: return
            tool_status = self.tool.run(self.env, coor, self.scale)
            if tool_status is True:
                print("Game status:", self.env.check_goal())
            elif tool_status is False:
                print("Tool failed")
            self.darea.queue_draw()
        if e.button == 2:
            self.mb_grasp = coor

    def check_goal(self):
        goals = self.env.cur_goal()
        constructed = tuple(self.env.objs_of_type(GeoObject))
        constr_basic = []
        constr_desired = []
        for obj in constructed:
            if any(goal.identical_to(obj) for goal in goals):
                constr_desired.append(obj)
            else:
                constr_basic.append(obj)
        goals_remaining = [
            goal for goal in goals
            if not any(goal.identical_to(obj) for obj in constr_desired)
        ]
        if len(goals_remaining) == 0:
            return True
        else:
            return False

    def run_llm_steps(self, steps, pass_at_k=50):
        if pass_at_k == 1:
            assert isinstance(steps, list)
            if not (isinstance(steps[0], list)):
                steps = [steps]
        else:
            assert isinstance(steps, list)
            assert (isinstance(steps[0], list))
        solved = 0
        for k in range(len(steps)):
            _ = Drawing(self.env)
            Gtk.main()
            for step in steps[k]:
                try:
                    self.env.add_and_run(
                        ConstructionProcess(step[0], step[1], self.env.generate_construction(self.env)[0]))
                except AttributeError:
                    # Reset Env instead of reloading level -- Much Faster #
                    self.env.steps = []
                    self.env.rand_steps = []
                    self.env.visible = set()
                    self.env.obj_to_movable = dict()
                    self.env.obj_num = 0
                    self.env.goal_index = 0
                    self.env.construction = None
                    self.env.construction_objects = None
                    self.env.construction_steps = None
                    continue
            if self.check_goal():
                solved += 1
            # Reset Env instead of reloading level -- Much Faster #
            self.env.steps = []
            self.env.rand_steps = []
            self.env.visible = set()
            self.env.obj_to_movable = dict()
            self.env.obj_num = 0
            self.env.goal_index = 0
            self.env.construction = None
            self.env.construction_objects = None
            self.env.construction_steps = None
        return len(steps), solved
