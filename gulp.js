var root = './js/fingerprint';
var temp = './.tmp/';
var config = {
	js: [root + '/static/**/*.js'],
	assets: [root + '/static/assets/*.*'],
	build: './dist/',
	index: root + '/index.html',
	root: root,
	temp: temp,
	optimized: {
		app: 'sdidlib.js',
	},
};

var gulp = require('gulp');
var babel = require('gulp-babel');
var uglify = require('gulp-uglify-es').default;
var $ = require('gulp-load-plugins')({ lazy: true});

gulp.task('assets', function() {
	log('copying asssts');
  
	return gulp
	  .src(config.assets)
	  .pipe(gulp.dest(config.build + 'static/assets'));
});

gulp.task('optimize', ['inject'], function() {
	log('Optimizing the js and html');
	var jsAppFilter = $.filter('**/' + config.optimized.app);
  
	return gulp
	  .src(config.index)
	  .pipe($.plumber())
	  .pipe(jsAppFilter)
	  .pipe(babel({presets: ['es2015']}))
	  .pipe(uglify())
	  .pipe(getHeader())
	  .pipe(jsAppFilter.restore())
	  .pipe(uglify()) // another option is to override wiredep to use min files
	  .pipe($.useref())
	  // Replace the file names in the html with rev numbers
	  .pipe($.revReplace())
	  .pipe(gulp.dest(config.build));
  });

gulp.task('build', ['assets', 'optimize'], function() {
	log('Building everything');
	var msg = {
	  title: 'gulp build',
	  subtitle: 'Deployed to the build folder',
	  message: 'Build Success`'
	};
	del(config.temp);
	log(msg);
	notify(msg);
});