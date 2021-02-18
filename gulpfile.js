var root = 'js/fingerprint';
var temp = './.tmp/';
var config = {
	jsDirec: root + '/static',
	js: [root + '/static/**/*.js'],
	assets: [root + '/static/assets/*.*'],
	build: './dist',
	index: root + '/index.html',
	root: root,
	temp: temp,
	optimized: {
		app: 'sdidlib.js',
	},
};
var del = require('del');
var gulp = require('gulp');
var order = require("gulp-order");
var concat = require('gulp-concat');
var minify = require('gulp-minify');
var rev = require('gulp-rev');
var babel = require('gulp-babel');
var uglify = require('gulp-uglify-es').default;
var $ = require('gulp-load-plugins')({ lazy: true});

gulp.task('assets', function() {
	console.log('copying asssts');
  
	return gulp
	  .src(config.assets)
	  .pipe(gulp.dest(config.build + '/static/assets'));
});

gulp.task('watch', function() {
	gulp.watch(config.js, ['pack-js']);
});

gulp.task('clean-js', function () {
    return del([
        config.build + '/js/*.js'
    ]);
});

gulp.task('pack-js', ['clean-js'], function () {    
    return gulp.src(config.js)
		.pipe(order([
			config.jsDirec + '/js/index.js',
			config.jsDirec + '/js/fontdetect.js',
			config.jsDirec + '/languages/languageDetector.js',
			config.jsDirec + '/js/advert.js',
			config.jsDirec + '/js/cookie.js',
			config.jsDirec + '/js/util.js',
			config.jsDirec + '/js/gl-matrix.js',
			config.jsDirec + '/js/gl-matrix.js',
			config.jsDirec + '/js/audio.js',
			config.jsDirec + '/js/detect-zoom.min.js',
			config.jsDirec + '/js/sha1.js',
			config.jsDirec + '/js/languageDetector.js',
			config.jsDirec + '/js/toServer.js',
			config.jsDirec + '/depth_texture/framework.js',
			config.jsDirec + '/depth_texture/meshes.js',
			config.jsDirec + '/depth_texture/webgl-nuke-vendor-prefix.js',
			config.jsDirec + '/depth_texture/webgl-texture-float-extension-shims.js',
			config.jsDirec + '/three/three.js',
			config.jsDirec + '/three/js/Detector.js',
			config.jsDirec + '/three/js/shaders/FresnelShader.js',
			config.jsDirec + '/three/js/loaders/DDSLoader.js',
			config.jsDirec + '/three/js/loaders/PVRLoader.js',
			config.jsDirec + 'three/lighting.js',
			config.jsDirec + '/three/bubbles.js',
			config.jsDirec + '/three/clipping.js',
			config.jsDirec + '/cube/no_texture.js',
			config.jsDirec + '/camera/camera.js',
			config.jsDirec + '/line/app.js',
			config.jsDirec + '/simpleLight/app.js',
			config.jsDirec + '/moreLight/app.js',
			config.jsDirec + '/twoTexturesMoreLight/app.js',
			config.jsDirec + '/transparent/app.js',
			config.jsDirec + '/video/video.js',
			config.jsDirec + '/canvas/canvas.js',
			config.jsDirec + '/texture/app.js',
			config.jsDirec + 'depth_texture/vsm-filtered-shadow.js',
			config.jsDirec + '/three/compressedTexture.js',
			config.jsDirec + '/js/loader.js'
		], { base: './' }))
        .pipe(concat('sdid_bundle.js'))
		/*.pipe(minify({
			ext: {
				min: '.js'
			}
		}))
		.pipe(uglify())*/
        .pipe(rev())
        .pipe(gulp.dest(config.build + '/js'))
        .pipe(rev.manifest('/rev-manifest.json', {
            merge: true
        }))
        .pipe(gulp.dest(config.build))
});


gulp.task('build', ['watch', 'assets', 'pack-js'], function() {
	console.log('Building everything');
	var msg = {
	  title: 'gulp build',
	  subtitle: 'Deployed to the build folder',
	  message: 'Build Success`'
	};
	del(config.temp);
	console.log(msg);
});