from distutils.core import setup
setup(
  name = 'stopp',        
  packages = ['stopp'],   
  version = '0.2',      
  license='MIT',        
  description = 'Safe Time Optimal Path Parametrization (STOPP) for serial robots',   
  author = 'Abdelrhman Bassiouny',                   
  author_email = 'Bassio@programmer.com',      
  url = 'https://github.com/AbdelrhmanBassiouny/stopp',   
  download_url = 'https://github.com/AbdelrhmanBassiouny/stopp/archive/v_02.zip',    
  keywords = ['robot trajectory', 'safe', 'jerk limited', 'moveit path support'],   
  install_requires=[            
          'numpy>=1.12.0',
      ],
  classifiers=[
    'Development Status :: 4 - Beta',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',  
    'Programming Language :: Python',
  ],
)
