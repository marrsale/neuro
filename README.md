# Neuro

This is an experimental neural network currently suited to creating a [multilayer perceptron](http://en.wikipedia.org/wiki/Multilayer_perceptron) with one hidden layer.  

Currently it is not very robust and I am in the process of adding some more generality and functionality so that it can be used in real applications (*Not recommended*).

Included in the repo is `main.rb` which contains a very basic example program which can be used to test the MLP.  To run the example, you must first
```bundle install```
to install the gems in the Gemfile, and then the program can be executed with
```bundle exec ruby main.rb```

Due to the randomness of the neurons when they are initialized, performance can be highly variable.  Here is some actual sample output from a run:

```
bundle exec ruby main.rb
For iteration #2000, error term is 0.003656866482768353.

.
.
.

For iteration #98000, error term is 2.2585318513659154e-06.
For input [1, 1]	[0.06164708960224026]
For input [1, 0]	[0.9544334325171219]
For input [0, 0]	[0.019713717703981025]
For input [0, 1]	[0.9532905467326417]
```

Note that as of this writing the only gem included is `pry` which is used to add breakpoints for debugging purposes.  If you don't wish to use this gem, simply comment out that line before you `bundle install`

This repo and its contents belong to [Alexander Marrs](github.com/marrsale), but can be used, copied, shared and modified by anyone for any ethical purpose as long as the attributions said author are left in the code.
___
###### COPYLEFT ORIGINAL AUTHOR ALEXANDER MARRS (github.com/marrsale / twitter.com/alx_mars)
