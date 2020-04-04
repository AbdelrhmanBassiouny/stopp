from distutils.core import setup
setup(
  name = 'stopp',         # How you named your package folder (MyLib)
  packages = ['stopp'],   # Chose the same as "name"
  version = '0.0',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'Safe Time Optimal Path Parametrization (STOPP) for serial robots',   # Give a short description about your library
  author = 'Abdelrhman Bassiouny',                   # Type in your name
  author_email = 'Bassio@programmer.com',      # Type in your E-Mail
  url = 'https://github.com/AbdelrhmanBassiouny/stopp',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/AbdelrhmanBassiouny/stopp/archive/v_00.tar.gz',    # I explain this later on
  keywords = ['robot trajectory', 'safe', 'jerk limited', 'moveit path support'],   # Keywords that define your package best
  install_requires=[            # I get to this in a second
          'numpy',
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3.5',
  ],
)