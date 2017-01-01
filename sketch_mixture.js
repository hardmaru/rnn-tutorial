// Basic Example of Unconditional Handwriting Generation.
var sketch = function( p ) { 
  "use strict";

  // variables we need for this demo

  var temperature = 0.65; // controls the amount of uncertainty of the model
  var Nmix = 20;
  var screen_width;
  var screen_height;

  var Nbar = 401;

  var base_pi = new Array(Nmix);
  var base_mu = new Array(Nmix);
  var base_sigma = new Array(Nmix);

  var pi = new Array(Nmix);
  var mu = new Array(Nmix);
  var sigma = new Array(Nmix);

  var ybar = new Array(Nbar);
  var xbar = new Array(Nbar);

  var factor = 5.0;

  var erf_constant = 1/Math.sqrt(2*Math.PI);

  var gaussian = function(x, mean, std) {
    var f = erf_constant / std;
    var p = -1/2;
    var c = (x-mean)/std;
    c *= c;
    p *= c;
    return f * Math.pow(Math.E, p);
  };

  var create_base_distribution = function() {
    var i, pi_sample;
    var pi_sum = 0;
    for (i=0;i<Nmix;i++) {
      pi_sample = Model.randf(0, 1);
      pi_sum += pi_sample;
      base_pi[i] = pi_sample;
      base_mu[i] = Model.randf(-factor*0.35, factor*0.35);
      base_sigma[i] = Model.randf(factor/20, factor/10);
    }
    for (i=0;i<Nmix;i++) {
      base_pi[i] /= pi_sum;
    }
    base_pi = adjust_temp(base_pi, 0.5);
    var deltax = factor/(Nbar-1);
    var startx = -deltax*(Nbar-1)/2;
    for (i=0;i<Nbar;i++) {
      xbar[i] = startx + i*deltax;
    }
  };

  var adjust_temp = function(z_old, temp) {
    var z = nj.array(z_old);
    var i;
    var x;
    //console.log("before="+z_old.get(0));
    for (i=z.shape[0]-1;i>=0;i--) {
      x = z.get(i);
      x = Math.log(x) / temp;
      z.set(i, x);
    }
    x = z.max();
    z = nj.subtract(z, x);
    z = nj.exp(z);
    x = z.sum();
    z = nj.divide(z, x);
    //console.log("after="+z.get(0));
    var z_array = new Array(Nmix);
    for (i=0;i<Nmix;i++) {
      z_array[i] = z.get(i);
    }
    return z_array;
  };

  var create_distribution = function() {
    var i, j;
    var x;
    var y;
    for (i=0;i<Nmix;i++) {
      pi[i] = base_pi[i];
      mu[i] = base_mu[i];
      sigma[i] = Math.max(temperature*base_sigma[i], factor/500);
    }
    pi = adjust_temp(pi, temperature);
    for (j=0;j<Nbar;j++) {
      x = xbar[j];
      y = 0;
      for (i=0;i<Nmix;i++) {
        y += pi[i]*gaussian(x, mu[i], sigma[i]);
      }
      ybar[j] = y;
    }
  };

  var draw_distribution = function() {
    var i, y;
    var delta = screen_width / Nbar;
    p.stroke(255, 165, 0);
    p.strokeWeight(1.0);
    for (i=0;i<Nbar;i++) {
      y = screen_height-screen_height*ybar[i]/4.0;
      p.line((i+0.5)*delta, screen_height, (i+0.5)*delta, y);
    }
  };

  var restart = function() {
    // reinitialize variables before calling p5.js setup.

    // make sure we enforce some minimum size of our demo
    screen_width = Math.max(window.innerWidth, 480);
    screen_height = Math.max(window.innerHeight, 320)/1.0;

    create_base_distribution();

  };

  var generate = function() {

    create_distribution();

    p.background(255);
    p.fill(255);
    // draws everything
    p.noStroke();
    p.fill(255, 165, 0, 128+127*temperature);
    p.rect(0, 0, screen_width*temperature, screen_height*0.10);
    p.textSize(40);
    p.text(Math.round(temperature*100)/100, screen_width*temperature+15, screen_height*0.10);

    draw_distribution();

  }

  p.setup = function() {
    restart(); // initialize variables for this demo
    p.createCanvas(screen_width, screen_height);
    p.frameRate(30);

    generate();
  };

  p.draw = function() {

  };

  var touched = function() {
    var mx = p.mouseX;
    if (mx >= 0 && mx < screen_width) {
      temperature = mx / screen_width;
      generate();
    }
  };

  p.touchMoved = touched;
  p.touchStarted = touched;

};
var custom_p5 = new p5(sketch, 'sketch');
