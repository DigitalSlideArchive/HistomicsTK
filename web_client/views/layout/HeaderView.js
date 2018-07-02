import View from '../View';
import HeaderAnalysesView from './HeaderAnalysesView';
import HeaderUserView from './HeaderUserView';
import HeaderImageView from './HeaderImageView';
import router from '../../router';

import headerTemplate from '../../templates/layout/header.pug';
import '../../stylesheets/layout/header.styl';

var HeaderView = View.extend({
    events: {
        'click #h-navbar-brand': function () {
            router.navigate('', {trigger: true});
        }
    },

    initialize(params) {
        this.settings = params.settings;
        return View.prototype.initialize.apply(this, arguments);
    },

    render() {
        this.$el.html(headerTemplate({
            brandName: this.settings.brandName,
            brandColor: this.settings.brandColor,
            bannerColor: this.settings.bannerColor
        }));

        this.$('a[title]').tooltip({
            placement: 'bottom',
            delay: {show: 300}
        });

        new HeaderUserView({
            el: this.$('.h-current-user-wrapper'),
            parentView: this
        }).render();

        new HeaderImageView({
            el: this.$('.h-image-menu-wrapper'),
            parentView: this
        }).render();

        new HeaderAnalysesView({
            el: this.$('.h-analyses-wrapper'),
            parentView: this
        }).render();

        return this;
    }
});

export default HeaderView;
