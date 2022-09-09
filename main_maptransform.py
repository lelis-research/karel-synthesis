from karel.world import World


if __name__ == '__main__':

    worlds = [
        '|  *|\n' +
        '|   |\n' +
        '|>  |\n' +
        '| * |\n' +
        '|   |',

        '|    |\n' +
        '| ***|\n' +
        '|  <*|\n' +
        '| ***|'
    ]

    for i, w_str in enumerate(worlds):

        world = World.from_string(w_str)
        world.center_and_pad(10, 10)
        print(world.to_string())