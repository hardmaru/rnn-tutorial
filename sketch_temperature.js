// Basic Example of Unconditional Handwriting Generation.
var sketch = function( p ) { 
  "use strict";

  // variables we need for this demo
  var dx, dy; // offsets of the pen strokes, in pixels
  var pen, prev_pen; // keep track of whether pen is touching paper
  var x, y; // absolute coordinates on the screen of where the pen is

  var rnn_state; // store the hidden states of rnn's neurons
  var pdf; // store all the parameters of a mixture-density distribution
  var temperature = 0.65; // controls the amount of uncertainty of the model

  var screen_width, screen_height; // stores the browser's dimensions
  var line_color;

  var restart = function() {
    // reinitialize variables before calling p5.js setup.

    // make sure we enforce some minimum size of our demo
    screen_width = Math.max(window.innerWidth, 480);
    screen_height = Math.max(window.innerHeight, 320);

    // start drawing from somewhere in middle of the canvas
    x = 50;
    y = screen_height/2;

    // initialize the scale factor for the model. Bigger -> large outputs
    Model.set_scale_factor(10.0);

    // initialize pen's states to zero.
    [dx, dy, prev_pen] = Model.zero_input(); // the pen's states

    // randomize the rnn's initial states
    rnn_state = Model.random_state();

    // define color of line
    line_color = p.color(255, 165, 0);

  };

  var generate = function() {

    p.noStroke();
    p.fill(255);
    p.rect(0, 0, screen_width, screen_height*0.105);

    p.fill(255, 165, 0, 128+127*temperature);
    p.rect(0, 0, screen_width*temperature/1.25, screen_height*0.10);
    p.textSize(40);

    p.text(Math.round(temperature*100)/100, screen_width*temperature/1.25+15, screen_height*0.10);

  }

  p.setup = function() {
    restart(); // initialize variables for this demo
    p.createCanvas(screen_width, screen_height);
    p.frameRate(60);
    p.background(255);
    p.fill(255);
    generate();
  };

  p.draw = function() {

    // using the previous pen states, and hidden state, get next hidden state 
    rnn_state = Model.update([dx, dy, prev_pen], rnn_state);

    // get the parameters of the probability distribution (pdf) from hidden state
    pdf = Model.get_pdf(rnn_state);

    // sample the next pen's states from our probability distribution
    [dx, dy, pen] = Model.sample(pdf, temperature);

    // only draw on the paper if the pen is touching the paper
    if (prev_pen == 0) {
      p.stroke(line_color);
      p.strokeWeight(2.0);
      p.line(x, y, x+dx, y+dy); // draw line connecting prev point to current point.
    }

    // update the absolute coordinates from the offsets
    x += dx;
    y += dy;

    // update the previous pen's state to the current one we just sampled
    prev_pen = pen;

    // if the rnn starts drawing close to the right side of the canvas, reset demo
    if (x > screen_width - 50) {
      restart();
      p.background(255); // fade out a bit.
      p.fill(255);
      generate();
    }

  };

  var touched = function() {
    var mx = p.mouseX;
    if (mx >= 0 && mx < screen_width) {
      temperature = 1.25*mx / screen_width;
      generate();
    }
  };

  p.touchMoved = touched;
  p.touchStarted = touched;

};
var custom_p5 = new p5(sketch, 'sketch');
