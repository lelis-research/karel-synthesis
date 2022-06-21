from dsl.production import Production

def main():

    config = dict()

    production = Production.default_karel_production()

    config['dsl'] = {}
    config['dsl']['num_agent_actions'] = len(production.get_actions()) + 1

if __name__ == '__main__':
    main()
