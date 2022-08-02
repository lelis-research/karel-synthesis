from karel.world_generator import WorldGenerator


if __name__ == '__main__':

    worldGen = WorldGenerator(69)
    world = worldGen.generate()
    print(world.to_string())