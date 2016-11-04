import View from '../View';
import HeaderUserView from './HeaderUserView';
import router from '../../router';

import headerTemplate from '../../templates/layout/header.pug';

var HeaderView = View.extend({
    events: {
        'click .g-app-title': function () {
            router.navigate('', {trigger: true});
        }
    },

    render: function () {
        this.$el.html(headerTemplate());

        this.$('a[title]').tooltip({
            placement: 'bottom',
            delay: {show: 300}
        });

        new HeaderUserView({
            el: this.$('.g-current-user-wrapper'),
            parentView: this
        }).render();
    }
});

export default HeaderView;
