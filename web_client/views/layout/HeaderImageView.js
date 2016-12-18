import events from '../../events';
import View from '../View';

import headerImageTemplate from '../../templates/layout/headerImage.pug';
import '../../stylesheets/layout/headerImage.styl';

var HeaderImageView = View.extend({
    events: {
        'click .h-open-image': function (evt) {
        }
    },

    initialize() {
        this.imageModel = null;
        this.listenTo(events, 'h:imageOpened', (model) => {
            this.imageModel = model;
            this.render();
        });
    },

    render() {
        var name = 'Open image...';
        if (this.imageModel) {
            name = this.imageModel.get('name');
        }
        this.$el.html(headerImageTemplate({name}));
        return this;
    }
});

export default HeaderImageView;
