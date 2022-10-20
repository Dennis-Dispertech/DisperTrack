from setuptools import setup, find_packages

setup(
    name='DisperTrack',
    version='1.0',
    url='https://www.dispertech.com',
    license='',
    author='Aquiles Carattino',
    author_email='carattino@dispertech.com',
    description='Software suit to track and analyse nanoparticles',
    include_package_data=True,
    packages=find_packages('.'),
    package_data={
        "": ['GUI/*.ui', 'GUI/Icons/*.svg', 'GUI/Icons/*.png']
        },
    entry_points = {'console_scripts':
        ['dispertech=dispertrack:start_analysis']
        },
    install_requires=[
        'pandas',
        ]
    )
